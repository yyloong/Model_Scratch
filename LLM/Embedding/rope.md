### RoPE
目标:$\langle f(q_m,m),f(k_n,n) \rangle = g(q_m,k_n,m-n)$
先考虑二维场景，利用复数,$f(q_m,m) = r(q_m,m)e^{\theta(q_m,m)}$
$f(k_n,n) = r(k_n,n)e^{\theta(k_n,n)}$
$g(q_m,k_n,m-n) = R(q_m,k_n,m-n)e^{\Theta(q_m,k_n,m-n)}$
有$r(q_m,m)r(k_n,n) = R(q_m,k_n,m,n)$
$\theta(q_m,m)-\theta(k_n,n) = \Theta(q_m,k_n,m-n)$
容易找到一组解
$r(q_m,m)= |q|,\theta(q_m,m) = m\theta + \theta_0$
考虑$\theta(q_m,0) = \theta(q)$
则
$$
f(q_m,m) = qe^{im\theta}=\begin{pmatrix}
cosm\theta & -sinm\theta \\
sinm\theta & cosm\theta
\end{pmatrix}
\begin{pmatrix}
q_0 \\
q_1
\end{pmatrix}
$$
利用内积的线性性质
$\langle q,k \rangle = \sum_{i=0}^{\frac{d}{2}-1} \langle q[i:i+2] , k[i:i+2] \rangle = \sum_{i=0}^{\frac{d}{2}-1}g(q[i:i+2],k[i:i+2],m-n) = G(q,k,m-n)$
同时考虑不同维度使用不同的$\theta$,参考绝对编码方案,$\theta_i = 10000^{-\frac{2i}{d}}$
矩阵非常稀疏，用逐元素相乘代替矩阵乘法加速计算
$f(q_m,m) = q \cdot cosm\theta + \hat{q} \cdot sinm\theta,\hat{q}$将奇数和偶数位置元素交换，奇数位置乘上$-1$

### 理解RoPE不同维度的频率特性
从$\theta_i$的定义来看,当$i$比较小时$\theta$很大，$m\theta$变化很快,是高频，反之为低频,高频对微小变化敏感，低频变化较慢，可以标注长距离的绝对位置信息

### RoPE不同维度使用不同$\theta$的原因
- 避免线性依赖(相同$\theta$相当于将高维特征进行了压缩)
- 唯一编码抗混叠,(否则序列长度大的时候会出现循环)
- 拥有远程衰减特性
- 利用不同维度的不同频率捕捉不同位置的信息

### 变长RoPE的改进
- 高频$\theta$在原长度内已经重复多次，无需再进行扩展
- 低频$\theta$在原长度内仍未满一个周期,需要将范围压缩到原长度所在的区间内

#### NTK-Aware
s为扩展倍数
修改base,让高频不怎么变化，低频发生全变化
$(b\cdot s^{\frac{d}{d-2}})^{\frac{-2i}{d}}$

#### YaRN
- 不修改base,修改$\theta$
- 高频不插值
- 低频全插值
- 中间段平滑过渡
$\alpha_i = clamp(\frac{\frac{L_{train}}{\lambda_i}-\beta_{slow}}{\beta_{fast}-\beta_{slow}},0,1)$
$\beta_{slow}$通常为1
$\theta_i^{YaRN} = \frac{\theta_i}{\gamma_i}$
$\gamma_i = (1-\alpha_i)\cdot s + \alpha_i \cdot 1$
- 插值会导致点积模长下降(注意力分布变平滑),需要引入全局缩放因子$\sqrt{t},t=0.1ln(s)+1$
$f_{YaRN}(x,m,\theta^{YaRN}) = f_{RoPE}(x,m,\theta^{RoPE}) \cdot \sqrt{t}$

