from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from .formatter import ChartFormatter
from .recorder import MicRecorder
from .transcriber import WhisperTranscriber


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="마이크 음성을 녹음해 Whisper로 인식하고 진료 차팅 텍스트로 출력합니다."
    )
    parser.add_argument(
        "--backend",
        choices=["api", "local"],
        default="api",
        help="음성인식 방식: api(OpenAI Whisper API, 기본값) 또는 local(오프라인 faster-whisper)",
    )
    parser.add_argument(
        "--stt-model",
        default="whisper-1",
        help="[--backend api] OpenAI 음성인식 모델 (예: whisper-1, gpt-4o-transcribe, gpt-4o-mini-transcribe)",
    )
    parser.add_argument(
        "--local-model-size",
        default="medium",
        help="[--backend local] faster-whisper 모델 크기 (tiny/base/small/medium/large-v3 등, 기본 medium)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="[--backend local] 실행 장치 (cpu 또는 cuda, 기본 cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        help="[--backend local] 연산 정밀도 (cpu는 int8 권장, cuda는 float16 권장)",
    )
    parser.add_argument("--language", default="ko", help="인식 언어 코드 (기본값: ko)")
    parser.add_argument(
        "--prompt",
        default=None,
        help="의료 용어 인식을 돕기 위한 힌트 (예: 자주 나오는 약품명, 진단명)",
    )
    parser.add_argument(
        "--soap",
        action="store_true",
        help="받아쓴 텍스트를 SOAP 형식 진료기록으로 정리합니다 (추가 API 호출 발생)",
    )
    parser.add_argument(
        "--chat-model",
        default="gpt-4o-mini",
        help="--soap 사용 시 정리에 사용할 채팅 모델",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="결과 텍스트를 저장할 파일 경로 (지정하지 않으면 화면에만 출력)",
    )
    parser.add_argument(
        "--keep-audio",
        action="store_true",
        help="녹음한 오디오 파일(wav)을 삭제하지 않고 recordings/ 폴더에 보관합니다",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    api_key = os.environ.get("OPENAI_API_KEY")
    needs_api_key = args.backend == "api" or args.soap
    if needs_api_key and not api_key:
        print(
            "OPENAI_API_KEY가 설정되어 있지 않습니다. .env 파일을 만들거나 "
            "환경변수로 지정하세요. (ChatGPT Plus 구독과 API 키는 별개입니다)\n"
            "API 키 없이 쓰려면 --backend local --soap 없이 실행하세요.",
            file=sys.stderr,
        )
        return 1

    recorder = MicRecorder()
    input("녹음을 시작하려면 Enter 키를 누르세요...")
    recorder.start()
    print("녹음 중입니다. 멈추려면 Enter 키를 누르세요...")
    input()
    audio = recorder.stop()

    if audio.shape[0] == 0:
        print("녹음된 오디오가 없습니다.", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_dir = Path.cwd() / "recordings" if args.keep_audio else Path(tempfile.gettempdir())
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_path = audio_dir / f"chart_{timestamp}.wav"
    recorder.save_wav(audio, str(audio_path))

    if args.backend == "local":
        try:
            from .local_transcriber import LocalWhisperTranscriber
        except ImportError:
            print(
                "로컬 백엔드를 쓰려면 faster-whisper가 필요합니다: "
                "pip install -r requirements-local.txt",
                file=sys.stderr,
            )
            return 1
        print(f"로컬 Whisper 모델을 불러오는 중... ({args.local_model_size}, {args.device})")
        transcriber = LocalWhisperTranscriber(
            model_size=args.local_model_size,
            device=args.device,
            compute_type=args.compute_type,
        )
    else:
        transcriber = WhisperTranscriber(api_key=api_key, model=args.stt_model)
    print("음성 인식 중...")
    transcript = transcriber.transcribe(audio_path, language=args.language, prompt=args.prompt)

    result = transcript
    if args.soap:
        print("차팅 텍스트로 정리 중...")
        formatter = ChartFormatter(api_key=api_key, model=args.chat_model)
        result = formatter.to_soap(transcript)

    print("\n----- 결과 -----\n")
    print(result)

    if args.output:
        args.output.write_text(result, encoding="utf-8")
        print(f"\n결과를 저장했습니다: {args.output}")

    if not args.keep_audio:
        audio_path.unlink(missing_ok=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
