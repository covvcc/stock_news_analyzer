# 진료 녹취 → 차팅 텍스트 변환기

마이크로 들은 진료 대화를 녹음해서 OpenAI 음성인식(Whisper)으로 한국어 텍스트로 옮기고,
필요하면 SOAP 형식의 진료기록으로 정리해 주는 CLI 도구입니다.

## 먼저 확인하세요: ChatGPT Plus ≠ OpenAI API

- ChatGPT **Plus** 구독(월 20불)은 chat.openai.com 앱 사용권한이고, 코드에서 호출하는
  **API**와는 별개의 결제입니다.
- 이 도구가 Whisper를 호출하려면 https://platform.openai.com 에서 **API 키**를 발급받고
  별도로 결제 수단(종량제)을 등록해야 합니다.
- API 키는 `.env` 파일에 `OPENAI_API_KEY=sk-...` 형태로 넣어주세요 (`.env.example` 참고).

## 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows는 .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # .env 안에 실제 API 키 입력
```

macOS는 마이크 접근 권한이 필요하고, Linux는 PortAudio 시스템 패키지
(`sudo apt install portaudio19-dev` 등)가 없으면 `sounddevice` 설치/실행이 실패할 수 있습니다.

## 사용법

```bash
python main.py
```

- Enter를 누르면 녹음이 시작되고, 다시 Enter를 누르면 녹음이 끝나고 바로 인식이 시작됩니다.
- 기본은 받아쓴 텍스트를 그대로 화면에 출력합니다.

### SOAP 형식 진료기록으로 정리하기

```bash
python main.py --soap --output chart.txt
```

`--soap`을 켜면 받아쓴 텍스트를 다시 GPT 모델에 보내 Subjective/Objective/Assessment/Plan
형식으로 정리합니다 (채팅 모델 추가 호출 → 비용이 더 듭니다).

### 자주 나오는 의료 용어 인식률 높이기

```bash
python main.py --prompt "고혈압, 당뇨병, 메트포르민, 암로디핀"
```

Whisper는 `--prompt`로 준 텍스트를 힌트로 사용해 비슷한 발음의 전문용어를 더 정확히 인식합니다.

### 옵션

| 옵션 | 설명 |
| --- | --- |
| `--backend` | `api`(OpenAI Whisper API, 기본값) 또는 `local`(오프라인 faster-whisper) |
| `--stt-model` | [`api`] 음성인식 모델 (기본 `whisper-1`, `gpt-4o-transcribe`/`gpt-4o-mini-transcribe`로 변경 가능) |
| `--local-model-size` | [`local`] faster-whisper 모델 크기 (`tiny`/`base`/`small`/`medium`/`large-v3`, 기본 `medium`) |
| `--device` | [`local`] 실행 장치 (`cpu` 기본, GPU 있으면 `cuda`) |
| `--compute-type` | [`local`] 연산 정밀도 (`cpu`는 `int8` 권장, `cuda`는 `float16` 권장) |
| `--language` | 인식 언어 코드 (기본 `ko`) |
| `--prompt` | 인식 힌트로 줄 텍스트 |
| `--soap` | 차팅용 SOAP 형식으로 정리 (이 옵션은 `--backend local`이어도 OpenAI API 호출 발생) |
| `--chat-model` | `--soap` 정리에 쓸 채팅 모델 (기본 `gpt-4o-mini`) |
| `--output` | 결과를 저장할 파일 경로 |
| `--keep-audio` | 녹음 파일(wav)을 삭제하지 않고 `recordings/`에 보관 |

## 로컬 Whisper로 실행하기 (API 키 불필요, 오프라인)

`faster-whisper`(CTranslate2 기반)를 백엔드로 지원합니다. 인터넷 없이, 건당 API 비용 없이,
음성 데이터를 외부로 전송하지 않고 로컬에서만 인식합니다.

```bash
pip install -r requirements-local.txt
python main.py --backend local --local-model-size medium
```

- 모델은 최초 실행 시 한 번 다운로드되어 캐시되고(`~/.cache/huggingface`), 이후에는 완전히 오프라인으로 동작합니다.
- `--soap`을 함께 쓰면 그 단계만 OpenAI API를 호출하므로 그때는 `OPENAI_API_KEY`가 필요합니다.
- GPU(NVIDIA + CUDA/cuDNN)가 있으면 `--device cuda --compute-type float16`으로 속도를 크게 높일 수 있습니다.

모델 크기별 대략적인 트레이드오프 (한국어 기준):

| 모델 | 속도 | 정확도 | 비고 |
| --- | --- | --- | --- |
| `tiny` / `base` | 매우 빠름 | 낮음 | 빠른 테스트용, 의료 차팅에는 부적합 |
| `small` | 빠름 | 보통 | CPU에서도 실용적 |
| `medium` | 보통 (기본값) | 좋음 | 정확도/속도 균형, CPU에서도 사용 가능하지만 다소 느림 |
| `large-v3` | 느림(CPU) / 보통(GPU) | 가장 좋음 | 전문용어가 많으면 권장, GPU 없으면 체감 지연이 큼 |

## 한국어 인식이 더 필요할 때 고려할 대안

의료 전문용어가 많은 환경에서 Whisper(API/로컬)만으로 부족하다면:

- **네이버 클로바 스피치 / Return Zero 등 한국어 특화 STT**: 커스텀 사전(단어 부스팅) 기능으로
  의료 용어 인식률을 높일 수 있으나 별도 API 신청·계약·요금 확인이 필요합니다.
- **Google Cloud Speech-to-Text**: 한국어 인식 품질이 좋고 커스텀 어휘(word boosting)를
  지원합니다.

`recorder.py`(녹음) / `transcriber.py`(인식) / `formatter.py`(차팅 정리)로 역할을 분리해
두었기 때문에, `transcriber.py`만 바꿔 끼우면 다른 STT 엔진으로 쉽게 교체할 수 있습니다.

## 개인정보/보안 참고

진료 대화에는 민감한 개인정보(PHI)가 포함됩니다. 클라우드 API(OpenAI 등)로 전송하면 해당
업체의 데이터 처리 정책이 적용되므로, 실제 임상 환경에 배포하기 전에 소속 기관의 개인정보
보호·컴플라이언스 요건을 반드시 확인하세요.
