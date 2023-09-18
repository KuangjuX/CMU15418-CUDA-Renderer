# CUDA Renderer Report

## CUDA Warm-Up 1: SAXPY

## CUDA Warm-UP 2: Parallel Prefix-Sum

## A Simple Circle Renderer

首先熟悉一下 `refRender.cpp` 的实现，`setup()` 首先在生成帧之前被调用，`refRender.cpp` 中什么都没有做。`render()` 被调用用于去渲染帧，并且负责在输出的图像中画圆圈。另一个主要的方法是 `advanceAnimation()`，它在渲染每个帧的时候也被调用，用于更新 circles 的位置和速度，在这个实验中我们不需要修改 `advanceAnimation()` 方法。

渲染每个帧的算法如下：
```
Clear image 
For each circle: 
    Update position and velocity 
For each circle: 
    Compute screen bounding box 
    For all pixels in bounding box: 
        Compute pixel center point 
        If center point is within the circle: 
            Compute color of circle at point 
            Blend contribution of circle into image for this pixel
```

Render 的一个重要细节是它渲染半透明的圆圈，任何一个像素的颜色都不是单个圆的颜色，而是混合与该像素
重叠的所有半透明圆的贡献的结果。渲染器通过 R，G，B，alpha 四元组用来表示圆的颜色。alpha = 1 用于表示完全不透明的圆，alpha = 0 用于表示完全透明的圆。如果希望在一个 ($P_r$, $P_g$, $P_b$) 的像素上面画 ($C_r$, $C_g$, $C_b$, $\alpha$)，render 可以执行如下计算：

$$
R_r = \alpha * C_r + (1.0 - \alpha) * P_r  
$$
$$
R_g = \alpha * C_g + (1.0 - \alpha) * P_g
$$
$$
R_b = \alpha * C_b + (1.0 - \alpha) * P_b
$$

需要注意在 Y 上的 X 与在 X 上的 Y 是不一样的。因此按照应用程序提供的顺序渲染圆非常重要。

### CUDA Renderer
目前 CUDA Renderer 中的实现可以并行计算所有输入圆，为每个 CUDA 线程分配一个输入圆。虽然此 CUDA Renderer 是 Circle Renderer 的正确数学实现，但它包含几个错误：

- Atomicity：所有图像更新操作必须是原子的。读取关键区域的 rgba 的值、将当前圆的贡献与当前图像进行混合、将其写入图像中。
- Order：Renderer 必须按图像的输入顺序对于像素进行操作。

想法：
- 从前到后生成拓扑序列。
- 从前到后渲染生成的 circle，不断渲染未重叠的像素点，使用 CUDA 并行处理，并用数组记录已经被渲染过的 circle。并使用一个二维数组来记录每次渲染过的像素点。