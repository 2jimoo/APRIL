# pilot5-4omini 실험 데이터

`docs/bte_pilot_report.pdf`의 4장(최종 파일럿 결과)과 5장(case study)에 쓴 원본 데이터입니다. 저장소 루트의 `data/`, `attack_feedback/`, `logs/`, `result/`는 `.gitignore` 대상이라, 같은 상대 경로 구조로 이 폴더에 복사해 두었습니다.

- 설정: NQ 5문항(`test1, 11, 16, 19, 20`), 스트림 3, 반복 5회, K=5, M=5, 모든 LLM 역할 `openai:gpt-4o-mini`
- 재현 명령: `docs/bte_pilot_report.pdf` 부록

## 구성

| 경로 | 내용 |
|---|---|
| `result/nq/robust_experiment/pilot5-4omini/report.{md,json}` | prevention rate 표 (보고서 4.2) |
| `data/nq/defense/pilot5-4omini-{bte,april}.json` | BTE 방어 문서와 그 재료가 된 사건 (Case 1~3의 방어 문서 원문) |
| `data/nq/targets.json` | 질문, 정답, 타깃 오답 (PoisonedRAG 공개 데이터) |
| `attack_feedback/nq/<공격>__pilot5-4omini-<코퍼스>/*.summary.json` | 질문별 오답 여부 `per_question` (보고서 4.3) |
| `attack_feedback/nq/<공격>__pilot5-4omini-<코퍼스>/k5_*.json` | 질문별 최고 점수 시도. RAG 응답, 오염 문서, 검색 순위 100개와 점수 (Case 1~4의 응답, top-5 구성, `[LATEST]` 순위) |
| `attack_feedback/.../*.eval_gpt-4o-mini.json` | RAGForensics 추적·제거 결과. `naive`는 학습 사건 수집용(보고서 4.1)입니다. `bte`·`april`은 prevention 기준으로 바꾸기 전 run에서 남은 참고용 파일입니다 |
| `logs/<공격>__pilot5-4omini-<코퍼스>_*.jsonl` | APRIL 반복별 전체 궤적 (improvement, 오염 문서, 응답, 점수, 루프 안 탐지 결과) |
| `logs/robust_pilot5-4omini.out` | 실험 실행 로그 |

`<코퍼스>`가 `naive`면 원본 코퍼스에 대한 학습 공격이고, `bte`·`april`이면 각 방어 코퍼스에 대한 테스트 공격입니다.

## case study별 위치

| Case | 질문 | 볼 파일 |
|---|---|---|
| 1 커버리지 | test20 | `data/nq/defense/pilot5-4omini-bte.json`에 test20 항목이 없음. `PAIR-adaptive-partial__pilot5-4omini-{bte,april}/k5_*.json`의 test20 |
| 2 반박형 문서 | test19 | `PAIR-adaptive-correction__pilot5-4omini-{bte,april}/k5_*.json`의 test19, 두 방어 문서의 test19 |
| 3 오답 개체 유입 | test11 | `PAIR-adaptive-correction__pilot5-4omini-april/k5_*.json`의 test11, `pilot5-4omini-april.json`의 test11 |
| 4 프록시 밀림 | test1 | 각 테스트 공격 `k5_*.json`의 test1에서 `context_texts` 중 `[LATEST]`가 든 항목의 순위와 `retrieval_scores` |
