from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parents[3] / 'outputs/report_figures/architecture'
OUT.mkdir(parents=True, exist_ok=True)
FONT = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
plt.rcParams.update({'svg.fonttype': 'none', 'axes.unicode_minus': False})
INK, MUTED, LINE = '#182c3e', '#536678', '#687c8d'
BLUE, BLUE_BG = '#246792', '#edf5fb'
ORANGE, ORANGE_BG = '#b66a23', '#fff6eb'
PURPLE, PURPLE_BG = '#7456a5', '#f4effa'
TEAL, TEAL_BG = '#287f78', '#edf7f5'


def canvas(width, height):
    fig, ax = plt.subplots(figsize=(width, height), dpi=110)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, width), ylim=(height, 0))
    ax.axis('off')
    return fig, ax


def label(ax, x, y, text, size=13, color=INK, ha='center', weight='normal', **kwargs):
    return ax.text(x, y, text, fontsize=size, fontproperties=FONT, color=color,
                   ha=ha, va='center', weight=weight, linespacing=1.45, **kwargs)


def box(ax, x, y, w, h, title, detail='', status='future', color=None, fill=None, size=14):
    color = color or (BLUE if status == 'built' else ORANGE)
    fill = fill or (BLUE_BG if status == 'built' else ORANGE_BG)
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=0.08',
                          lw=1.45, edgecolor=color, facecolor=fill,
                          linestyle='solid' if status == 'built' else (0, (5, 3)), zorder=3)
    ax.add_patch(patch)
    if detail:
        title_text = label(ax, x+w/2, y+h*0.30, title, size=size, zorder=4)
        detail_text = label(ax, x+w/2, y+h*0.69, detail, size=size-2, color=MUTED, zorder=4)
        checks.extend([(patch, title_text), (patch, detail_text)])
    else:
        checks.append((patch, label(ax, x+w/2, y+h/2, title, size=size, zorder=4)))
    return patch


def arrow(ax, points, color=LINE, dashed=False, lw=1.55):
    xs, ys = zip(*points)
    style = (0, (4, 3)) if dashed else 'solid'
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, linestyle=style, zorder=2)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                 mutation_scale=14, color=color, lw=lw, linestyle=style, zorder=2))


def save(fig, name):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for patch, text in checks:
        pb, tb = patch.get_window_extent(renderer), text.get_window_extent(renderer)
        assert tb.x0 >= pb.x0+3 and tb.x1 <= pb.x1-3, (name, text.get_text(), 'horizontal overflow')
        assert tb.y0 >= pb.y0+3 and tb.y1 <= pb.y1-3, (name, text.get_text(), 'vertical overflow')
    fig.savefig(OUT / f'{name}.png', dpi=220, facecolor='white')
    fig.savefig(OUT / f'{name}.svg', facecolor='white')
    fig.savefig(OUT / f'{name}-preview.png', dpi=90, facecolor='white')
    plt.close(fig)


checks = []
fig, ax = canvas(20, 15.4)
label(ax, .7, .50, '양손 EE pose 기반 계층적 양팔 조작 구조', 23, ha='left')
label(ax, .7, 1.05, 'VR·IK·공통 실행 기반에서 VLA·Planning·Progress Model로 확장', 14, color=MUTED, ha='left')
box(ax, 14.0, .4, 2.2, .50, '구현 기반', status='built', size=11)
box(ax, 16.55, .4, 2.8, .50, '후속 개발', size=11)

label(ax, .7, 1.65, '01  상태 인지와 스킬 선택', 14, ha='left', color=MUTED)
box(ax, .7, 2.1, 5.15, 1.22, '시각·촉각 관측', '영상 2D tracking + 접촉·압착 정보')
box(ax, 7.15, 2.1, 5.10, 1.22, 'Progress Model', '현재 작업 단계·진행 상태 추정')
box(ax, 13.55, 2.1, 5.80, 1.22, '상위 행동모델', 'VLA·Planning skill 선택·전환')
arrow(ax, [(5.85, 2.71), (7.15, 2.71)])
arrow(ax, [(12.25, 2.71), (13.55, 2.71)])

label(ax, .7, 4.03, '02  시연 학습과 작업별 동작 생성', 14, ha='left', color=MUTED)
box(ax, .7, 4.52, 5.15, 1.50, '착용형 UMI 시연', '사람의 파지점·자연스러운 궤적\n접근·적재 데이터 확보 및 학습')
box(ax, 7.15, 4.52, 5.10, 1.50, 'VLA skill', '접근 · 목표 지점 적재', color=TEAL, fill=TEAL_BG)
box(ax, 13.55, 4.52, 5.80, 1.50, 'Planning skill', '접촉면 정렬 · 압착·파지 · 이송')
arrow(ax, [(5.85, 5.27), (7.15, 5.27)], color=TEAL)
label(ax, 6.49, 5.04, '학습', 10, color=TEAL)
arrow(ax, [(16.45, 3.32), (16.45, 3.72), (9.7, 3.72), (9.7, 4.52)])
arrow(ax, [(16.45, 3.72), (16.45, 4.52)])

box(ax, .7, 6.98, 5.15, 1.05, 'VR 원격조작', '상대좌표 보정 · 연속 입력 보간', status='built')
box(ax, 7.15, 6.98, 12.2, 1.05, '공통 action space: 양손 EE pose 시퀀스', '왼손·오른손 목표 위치 및 자세', status='built')
arrow(ax, [(5.85, 7.5), (7.15, 7.5)], color=BLUE)
arrow(ax, [(9.7, 6.02), (9.7, 6.98)], color=TEAL)
arrow(ax, [(15.05, 6.02), (15.05, 6.98)], color=ORANGE)

ax.add_patch(Rectangle((.7, 8.65), 18.65, 2.72, facecolor='#f8fafc', edgecolor='#d7e1e8', lw=1, zorder=0))
label(ax, 1.02, 8.99, '03  공통 IK·궤적 계획·정책 실행 기반', 14, ha='left', color=BLUE)
box(ax, 1.02, 9.48, 4.82, 1.25, '인간 자세 기반 cost 예측', '관절 구성의 선호 비용 (prior)', status='built', color=PURPLE, fill=PURPLE_BG, size=13)
box(ax, 6.46, 9.48, 3.66, 1.25, 'IK solver', 'EE 추종 + 자세 cost\n이전 관절 해와의 연속성', status='built', size=13)
box(ax, 10.75, 9.48, 4.25, 1.25, '궤적 계획·유효성 검사', '관절 범위·속도·가속도\n이동 구간·보정 후 충돌 검사', status='built', size=13)
box(ax, 15.63, 9.48, 3.4, 1.25, '로봇 제어 인터페이스', '검사된 관절 궤적 전달', status='built', size=12)
arrow(ax, [(8.29, 8.03), (8.29, 9.48)], color=BLUE)
arrow(ax, [(5.84, 10.105), (6.46, 10.105)], color=PURPLE)
arrow(ax, [(10.12, 10.105), (10.75, 10.105)], color=BLUE)
arrow(ax, [(15, 10.105), (15.63, 10.105)], color=BLUE)
label(ax, 10, 11.04, '몸통 기준 양팔 IK  ·  독립 허리 회전  ·  하체 균형 제어와 연계', 12, color=MUTED)

box(ax, 14.0, 11.92, 5.35, 1.15, '접촉·압착 제어', 'Planning 조건 + 촉각 피드백')
arrow(ax, [(19.35, 5.27), (19.73, 5.27), (19.73, 12.495), (19.35, 12.495)], color=ORANGE)
label(ax, 19.83, 7.75, '접촉·압착 조건', 10, color=ORANGE, rotation=90)
box(ax, 7.15, 13.72, 8.25, 1.10, '휴머노이드 양팔 작업 수행', '시각·촉각·행동 데이터 취득', status='built', color=LINE, fill='#f3f5f7')
arrow(ax, [(17.33, 10.73), (17.33, 11.62), (10.3, 11.62), (10.3, 13.72)], color=BLUE)
label(ax, 10.54, 12.28, '관절 명령', 11, color=BLUE, ha='left')
arrow(ax, [(16.68, 13.07), (16.68, 14.27), (15.4, 14.27)], color=ORANGE)
arrow(ax, [(7.15, 14.27), (.26, 14.27), (.26, 2.71), (.7, 2.71)], color=LINE)
label(ax, 3.85, 14.0, '작업 관측 피드백', 12, color=MUTED)
label(ax, .7, 15.11, '구성도는 제공된 개발 현황과 후속 설계를 표현함. 박스 테두리: 실선 = 구현 기반, 점선 = 후속 개발.', 10, color=MUTED, ha='left')
save(fig, 'figure-1-hierarchical-architecture')

checks = []
fig, ax = canvas(20, 10.5)
label(ax, .7, .5, '작업 단계별 스킬 분담과 데이터 확보 전략', 23, ha='left')
label(ax, .7, 1.07, '후속 개발 구상  |  Progress Model이 상태를 추정하고, 상위 행동모델이 스킬 전환을 결정', 14, color=MUTED, ha='left')
box(ax, .7, 1.73, 18.65, .94, '시각 2D tracking + 촉각  →  Progress Model  →  단계별 스킬 선택·전환', size=15)

xs = [.7, 4.53, 8.36, 12.19, 16.02]
stages = [
    ('01  접근', 'VLA skill', '사람 시연의 자연스러운\n양손 접근 동작', TEAL, TEAL_BG),
    ('02  접촉면 정렬', 'Planning skill', '물체 기준 접촉 위치와\n양손 방향 정렬', ORANGE, ORANGE_BG),
    ('03  압착·파지', 'Planning skill', '균형 있는 접촉과\n안정적인 파지력 형성', ORANGE, ORANGE_BG),
    ('04  이송', 'Planning skill', '접촉 위치·상대 자세·\n압착 상태 유지', ORANGE, ORANGE_BG),
    ('05  적재', 'VLA skill', '목표 지점에 맞춘\n배치 동작 생성', TEAL, TEAL_BG),
]
for i, (x, (title, skill, detail, color, fill)) in enumerate(zip(xs, stages)):
    label(ax, x+1.66, 3.16, title, 15)
    box(ax, x, 3.60, 3.33, 1.63, skill, detail, color=color, fill=fill, size=14)
    arrow(ax, [(x+1.66, 2.67), (x+1.66, 2.95)], color=LINE)
    if i < 4:
        arrow(ax, [(x+3.33, 4.415), (xs[i+1], 4.415)])

box(ax, 8.36, 5.68, 10.99, .92, 'Planning 파지 유지  |  파지 형성 → 이송 → 적재 중 필요한 구간까지 연계', size=13)
arrow(ax, [(10.02, 5.23), (10.02, 5.68)], color=ORANGE)
arrow(ax, [(13.85, 5.23), (13.85, 5.68)], color=ORANGE)
arrow(ax, [(17.68, 5.23), (17.68, 5.68)], color=ORANGE)
label(ax, .7, 6.22, '데이터 확보와 학습', 15, color=MUTED, ha='left')

box(ax, .7, 7.02, 3.33, .95, 'UMI 시연 데이터', '접근 VLA 학습', color=TEAL, fill=TEAL_BG, size=12)
box(ax, 4.53, 7.02, 10.99, .95, 'Planning을 적용한 로봇 수행 데이터', '정렬 · 압착 · 파지 유지 · 이송', size=13)
box(ax, 16.02, 7.02, 3.33, .95, 'UMI 시연 데이터', '적재 VLA 학습', color=TEAL, fill=TEAL_BG, size=12)
box(ax, .7, 8.72, 11.00, .95, '전 과정 시각·촉각·행동 데이터 축적', '작업 단계와 수행 스킬 정보를 연계', size=13)
box(ax, 13.0, 8.72, 6.35, .95, 'End-to-End VLA 통합 학습', '모듈별 조작을 전체 작업으로 확장', color=TEAL, fill=TEAL_BG, size=13)
arrow(ax, [(2.365, 7.97), (2.365, 8.72)], color=TEAL)
arrow(ax, [(10.025, 7.97), (10.025, 8.72)], color=ORANGE)
arrow(ax, [(17.685, 7.97), (17.685, 8.34), (8.4, 8.34), (8.4, 8.72)], color=TEAL)
arrow(ax, [(11.70, 9.195), (13.0, 9.195)], color=TEAL)
label(ax, .7, 10.12, '모든 스킬의 EE 목표는 공통 IK·제약 검사 계층으로 전달됨. 단계 전환 조건과 접촉·압착 제어의 구현·통합은 후속 개발 항목임.', 11, color=MUTED, ha='left')
save(fig, 'figure-2-skills-and-learning')
print('Created 2 figures: editable SVG and 220 dpi PNG. All box labels fit.')
