import anthropic
import json

# 대화 내역에서 파일 ID 찾기
# (실제로는 대화 ID나 메시지 ID가 필요합니다)
client = anthropic.Anthropic()

# 예시: 응답에서 파일 정보 파싱
response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=1024,
    messages=[
        {"role": "user", "content": "이전에 생성한 파일 목록을 보여줘"}
    ]
)

print(response.content)
