# Git 작업 순서 (팀원용)

실행 위치: 터미널 (Mac: 터미널 앱 / Windows: Git Bash)

```bash
# 1) 처음 한 번 — 레포 받기
git clone https://github.com/ten-infosec/oz-cancer-subtype.git
cd oz-cancer-subtype

# 2) 내 브랜치 만들기 (이름-주제)
git switch -c feat/myungseok-mutation-types

# 3) 작업 후 매일 커밋 + 푸시
git add notebooks/ logs/
git commit -m "feat: 변이 종류별 개수 파생변수 실험 (CV 0.3xx)"
git push -u origin feat/myungseok-mutation-types

# 4) GitHub에서 Pull Request → 팀장 확인 후 main 머지
# 5) 다음 날 시작 전 main 최신화
git switch main && git pull
```

규칙
- `data/`와 `*.csv`는 .gitignore로 막혀 있음 — 실수로도 안 올라감
- 노트북은 출력 셀 포함해도 됨 (결과 확인용). 단 용량 50MB 넘으면 출력 지우고 커밋
- 커밋 메시지 앞에 `feat:` `exp:` `fix:` `docs:` 중 하나
