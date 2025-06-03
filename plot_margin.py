import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'

# parameter configuration
g = 9.81             # gravity acceleration [m/s^2]
h = 0.25             # brick height [m]
w = 0.247            # brick width[m]

# Number of bricks stacked
Nw = 1              # horizontally
Nh = 5              # vertically
friction_coefficient = None
# moment contributions
sum_w = sum([2*j - 1 for j in range(1, Nw + 1)])
sum_h = sum([2*k - 1 for k in range(1, Nh + 1)])
sum_friction = sum([k - 1 for k in range(1, Nh + 1)])

a_crit = g * sum_w * w * Nh / (sum_h * h * Nw)
if friction_coefficient is not None:
    a_crit = a_crit + 2 * sum_friction * friction_coefficient * g / sum_h
print('Critical Acceleration:', a_crit)

v_vals = np.linspace(0, 1.5, 200)          # lateral Velocity [m/s]
r_vals = np.linspace(0.0, 5.0, 200)        # Turning Radius [m]

V, R = np.meshgrid(v_vals, r_vals)
a_actual = V**2 / (R + 1e-6)                       # Actual centrifugal acceleration

# Stability Margin
margin = (a_crit - a_actual) / a_crit

plt.figure(figsize=(10, 6))
cmap = plt.cm.RdYlGn
levels = np.linspace(0, 1, 21)
contour = plt.contourf(V, R, margin, levels=levels, cmap=cmap, extend='both')
cbar = plt.colorbar(contour, label="Stability Margin")

# plt.title("Stability Margin of Autonomous Forklift (ZMP-based)")
plt.xlabel("Velocity [m/s]")
plt.ylabel("Turning Radius [m]")
plt.grid(True)

# Guideline line of stability and instability
plt.contour(V, R, margin, levels=[0], colors='black', linewidths=1.5)

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# パラメータ設定
g = 9.81                  # 重力加速度 [m/s^2]
W = 1.0                   # 全体幅 [m]
H = 1.25                  # 全体高さ [m]
b = 0.247                 # カラム幅 [m]
h = 1.25                  # カラム高さ [m]

# 臨界加速度の計算
a_crit_full = g * W / H
a_crit_column = g * b / h
a_crit_friction = a_crit

# 加速度範囲（0 〜 転倒域含む）
a_vals = np.linspace(0, max(a_crit_full, a_crit_column, a_crit_friction) * 1.2, 300)

# 安定性マージンの計算
margin_full = (a_crit_full - a_vals) / a_crit_full
margin_column = (a_crit_column - a_vals) / a_crit_column
margin_friction = (a_crit_friction - a_vals) / a_crit_friction

# 可視化
plt.figure(figsize=(10, 6))
plt.plot(a_vals, margin_full, label="Full Stack Margin", linewidth=2)
plt.plot(a_vals, margin_column, label="Edge Column Margin", linewidth=2, linestyle='--')
plt.plot(a_vals, margin_friction, label="Edge Column Margin with Friction", linewidth=2, linestyle=':')
plt.axhline(0, color='black', linestyle=':', label="Tipping Threshold")

# plt.title("Stability Margin vs. Lateral Acceleration")
plt.xlabel("Lateral Acceleration [m/s²]")
plt.ylabel("Stability Margin")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
