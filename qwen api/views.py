from rest_framework import status
from rest_framework.response import Response
from rest_framework.views import APIView

from qwen import QwenError, QwenUnavailable, ask_qwen
from serializers import LandmarkGuideRequestSerializer


class TouristGuideView(APIView):
    """Turn a landmark recognition result into a local Qwen visitor briefing."""

    def post(self, request):
        serializer = LandmarkGuideRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {
                    "success": False,
                    "errors": serializer.errors,
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        landmark_result = serializer.validated_data
        try:
            briefing = ask_qwen(landmark_result)
        except QwenUnavailable as exc:
            return Response(
                {
                    "success": False,
                    "errors": {"qwen": [str(exc)]},
                },
                status=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        except QwenError as exc:
            return Response(
                {
                    "success": False,
                    "errors": {"qwen": [str(exc)]},
                },
                status=status.HTTP_502_BAD_GATEWAY,
            )

        return Response(
            {
                "success": True,
                "source": landmark_result,
                "model": briefing["model"],
                "parsed": briefing["parsed"],
                "guide": briefing["guide"],
            },
            status=status.HTTP_200_OK,
        )
