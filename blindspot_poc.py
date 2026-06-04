# -*- coding: utf-8 -*-
"""
「外し方」には方向がある ― 人と機械の見逃しPoC（合成データ）

外観検査の誤りは、2種類ある。
  ・過検出（良品を「不良」と判定）… 良品を捨てるだけ。安全側の誤り。
  ・見逃し（不良を「良品」と判定）… 不良が世に出る。危険側の誤り。

本当に怖いのは、後者だ。良品を捨てるのは、もったいないが事故にはならない。
だが、不良を良品として通せば、それは市場に出て、事故や被害につながる。
航空機の部品なら、なおさらだ。

このPoCは、人と機械が「どちらの方向に外すか」を合成データで示す。
  ・人：迷ったとき、安全側（過検出）に倒れやすい。「怪しければ弾く」。
        ただし疲れると、後半は危険側の見逃しも増える。
  ・機械：よくあるキズは安定。だが学習にない想定外のキズを、
        危険側（見逃し）に、しかも自信ありげに通す。

点数（全体精度）ではなく、"どちらの方向に外すか" が主題。

※ 実データは一切使っていない。すべて合成データ。原理確認の第一歩。
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for path in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
             "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf"]:
    try:
        fm.fontManager.addfont(path)
    except Exception:
        pass
matplotlib.rcParams["font.family"] = ["Noto Sans CJK JP", "IPAGothic", "sans-serif"]
matplotlib.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)

# 2000個の部品を検査する。一定割合が「本当は不良」。
N = 2000
idx = np.arange(N)
is_defect = rng.random(N) < 0.15          # 15%が本当の不良
# 不良のうち一部は「想定外の形」（学習データに乏しい）
is_rare_defect = is_defect & (rng.random(N) < 0.20)  # 不良の2割が想定外

# 判定結果を「良品/不良」で出す。真偽と突き合わせて4分類する：
#   見逃し(miss)   = 本当は不良なのに「良品」と判定 ← 危険側（最も怖い）
#   過検出(false)  = 本当は良品なのに「不良」と判定 ← 安全側
#   正検出/正良品  = 正しい判定

def classify(call_defect):
    """call_defect: その部品を『不良』と判定したか(True/False)"""
    miss = is_defect & (~call_defect)      # 不良を見逃した（危険側）
    over = (~is_defect) & call_defect      # 良品を弾いた（安全側）
    return miss, over

# ---- 人の判定モデル ----
# 「怪しければ弾く」傾向（安全側に倒れる）→ 良品もやや多めに弾く（過検出が出る）。
# 不良はよく捕まえるが、疲労で後半は見逃しも増える。種類（想定外か）には鈍感。
fatigue = 0.10 * (idx / N)                          # 後半ほど見逃しやすい
human_call = np.zeros(N, dtype=bool)
# 良品を弾く確率（安全側のクセ）：一定して高め
human_call[~is_defect] = rng.random((~is_defect).sum()) < 0.08
# 不良を不良と捕まえる確率：高いが、疲労で少し落ちる（想定外でも違和感で拾える）
p_catch_defect = 0.93 - fatigue[is_defect]
human_call[is_defect] = rng.random(is_defect.sum()) < p_catch_defect

# ---- 機械の判定モデル ----
# 良品はめったに弾かない（過検出が少ない＝歩留まりは良い）。
# よくある不良はほぼ確実に捕まえる。だが想定外の不良を大きく見逃す（危険側）。
machine_call = np.zeros(N, dtype=bool)
machine_call[~is_defect] = rng.random((~is_defect).sum()) < 0.02   # 過検出は少ない
typ_defect = is_defect & (~is_rare_defect)
machine_call[typ_defect] = rng.random(typ_defect.sum()) < 0.98     # 典型不良はほぼ捕捉
machine_call[is_rare_defect] = rng.random(is_rare_defect.sum()) < 0.42  # 想定外は半分以上見逃す

human_miss, human_over = classify(human_call)
mach_miss, mach_over = classify(machine_call)

n_def = is_defect.sum()
n_good = (~is_defect).sum()
n_rare = is_rare_defect.sum()

print("=" * 64)
print(" 外し方には方向がある ― 人と機械の見逃しPoC（合成データ）")
print("=" * 64)
print(f"\n  部品 {N}個（うち本当の不良 {n_def}個、その中で想定外 {n_rare}個）")

print("\n■ 危険側の誤り＝『不良を良品として通した』数（少ないほど安全）")
print(f"    人   {human_miss.sum():3d} 個（不良の {100*human_miss.sum()/n_def:.0f}%）")
print(f"    機械 {mach_miss.sum():3d} 個（不良の {100*mach_miss.sum()/n_def:.0f}%）")

print("\n■ そのうち『想定外の不良』を通した数")
print(f"    人   {(human_miss & is_rare_defect).sum():3d} / {n_rare}  ({100*(human_miss&is_rare_defect).sum()/n_rare:.0f}%)")
print(f"    機械 {(mach_miss & is_rare_defect).sum():3d} / {n_rare}  ({100*(mach_miss&is_rare_defect).sum()/n_rare:.0f}%)")

print("\n■ 安全側の誤り＝『良品を不良として弾いた』数（多いと無駄だが事故にならない）")
print(f"    人   {human_over.sum():3d} 個（良品の {100*human_over.sum()/n_good:.0f}%）")
print(f"    機械 {mach_over.sum():3d} 個（良品の {100*mach_over.sum()/n_good:.0f}%）")

print("\n  ※ 機械は良品をあまり弾かない（歩留まりは良い）。")
print("     だが、いちばん怖い『不良を通す』を、想定外で大量にやる。")
print("     人は良品を弾きすぎる代わりに、危険側には倒れにくい。")
print("=" * 64)

# ============ 作図 ============
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.2))

# 左：誤りの「方向」を左右に振り分けた図
# 左向き(マイナス)＝安全側（過検出）、右向き(プラス)＝危険側（見逃し）
labels = ["人", "機械"]
over_vals = [100*human_over.sum()/n_good, 100*mach_over.sum()/n_good]      # 安全側
miss_vals = [100*human_miss.sum()/n_def, 100*mach_miss.sum()/n_def]        # 危険側
y = np.arange(2)
axL.barh(y, [-v for v in over_vals], color="#90A4AE", label="安全側：良品を弾く（過検出）")
axL.barh(y, miss_vals, color="#C62828", label="危険側：不良を通す（見逃し）")
axL.set_yticks(y); axL.set_yticklabels(labels, fontsize=12)
axL.axvline(0, color="black", lw=0.8)
axL.set_xlabel("← 安全側の誤り（％）　　　危険側の誤り（％）→")
axL.set_title("どちらの方向に外すか", fontsize=12)
axL.legend(loc="lower right", fontsize=9)
axL.grid(alpha=0.25, axis="x")
for yi, v in zip(y, over_vals):
    axL.text(-v-0.4, yi, f"{v:.0f}%", ha="right", va="center", fontsize=10, color="#546E7A")
for yi, v in zip(y, miss_vals):
    axL.text(v+0.4, yi, f"{v:.0f}%", ha="left", va="center", fontsize=10, color="#C62828", fontweight="bold")
axL.set_xlim(-12, 20)

# 右：いちばん怖い「不良を通す」を、よくある不良 vs 想定外の不良 で分解
cats = ["よくある不良", "想定外の不良"]
human_bytype = [
    100*(human_miss & ~is_rare_defect).sum()/max((is_defect&~is_rare_defect).sum(),1),
    100*(human_miss & is_rare_defect).sum()/max(n_rare,1)]
mach_bytype = [
    100*(mach_miss & ~is_rare_defect).sum()/max((is_defect&~is_rare_defect).sum(),1),
    100*(mach_miss & is_rare_defect).sum()/max(n_rare,1)]
x = np.arange(2); w = 0.35
axR.bar(x - w/2, human_bytype, w, color="#1565C0", label="人")
axR.bar(x + w/2, mach_bytype, w, color="#C62828", label="機械")
axR.set_xticks(x); axR.set_xticklabels(cats, fontsize=11)
axR.set_ylabel("不良を見逃した割合（％）")
axR.set_ylim(0, 70)
axR.set_title("『不良を通す』のは、機械の想定外で起きる", fontsize=12)
axR.legend(fontsize=10)
axR.grid(alpha=0.25, axis="y")
for xi, v in zip(x - w/2, human_bytype):
    axR.text(xi, v+1.5, f"{v:.0f}%", ha="center", fontsize=10, color="#1565C0", fontweight="bold")
for xi, v in zip(x + w/2, mach_bytype):
    axR.text(xi, v+1.5, f"{v:.0f}%", ha="center", fontsize=10, color="#C62828", fontweight="bold")

fig.suptitle("同じ「間違い」でも、安全側に外すか・危険側に外すかが違う", fontsize=14)
fig.tight_layout()
fig.savefig("blindspot.png", dpi=130)
print("\n図を出力：blindspot.png")
