"""
benchmark/

교수님 피드백(2026-09) 대응용 "메모리 특성 측정" 실험 코드.
src/pipeline, src/adapters(이미 GPU 검증 완료된 4개 mode)는 전혀 건드리지 않고,
이 폴더 안에서만 새 실험 인프라를 추가한다 - 기존 검증된 경로에 회귀를 만들지
않기 위함(이 브랜치의 설계 원칙, benchmark/README.md 참고).
"""
