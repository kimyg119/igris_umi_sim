#!/usr/bin/env bash
# NVIDIA RM 락업(머신 프리즈) 완화 — 영구 적용. 실행: sudo bash apply_freeze_fix.sh
#
# 진단(2026-08-31, 메모리 machine-freeze-nvidia-rm-lockup.md): Isaac 학습 중 프리즈의 원인은
# 580.95.05 드라이버의 RM ioctl NULL-deref (RIP 0x0, nvidia_unlocked_ioctl→rm_ioctl→_nv0561xx).
# python 스레드가 RM 전역 스핀락을 쥔 채 IRQ 끄고 죽어 전 CPU 하드락업 → 전원버튼만 남음.
# 8192-env Isaac 기동이 2회 연속(16:47, 16:51) 이를 촉발.
set -e
[ "$(id -u)" = 0 ] || { echo "sudo 로 실행하세요: sudo bash $0"; exit 1; }

# [1/2] oops/하드락업 → 20초 뒤 자동 리부팅 (수동 전원차단 제거). 즉시 + 영구.
cat > /etc/sysctl.d/99-panic-on-oops.conf <<'EOF'
# NVIDIA 580.95.05 RM NULL-deref -> 스핀락 데드락 완화 (2026-08-31)
# oops 방치 시 RM 락 미해제로 전 CPU 하드락업 -> 수동 전원차단 강요.
# panic 으로 승격해 20초 뒤 자동 리부팅.
kernel.panic_on_oops = 1
kernel.panic = 20
kernel.hardlockup_panic = 1
EOF
sysctl --system >/dev/null
echo "[1/2] panic-on-oops 적용됨(즉시+영구): panic_on_oops=$(sysctl -n kernel.panic_on_oops) panic=$(sysctl -n kernel.panic) hardlockup_panic=$(sysctl -n kernel.hardlockup_panic)"

# [2/2] GSP 펌웨어 오프 — 크래시 프레임이 GSP 오프로드 RM 경로. proprietary(nvidia-dkms-580) 전용
# (open 커널모듈은 GSP 필수라 불가). 다음 리부팅부터 적용. 원복 = 파일 삭제 + update-initramfs -u.
cat > /etc/modprobe.d/nvidia-gsp-off.conf <<'EOF'
options nvidia NVreg_EnableGpuFirmware=0
EOF
update-initramfs -u
echo "[2/2] GSP off 등록됨 — 다음 리부팅부터. 적용 확인: grep EnableGpuFirmware /proc/driver/nvidia/params (0 이어야 함)"
echo
echo "권장: 지금 리부팅해서 GSP off 까지 활성화한 뒤 무거운 Isaac 작업 재개."
