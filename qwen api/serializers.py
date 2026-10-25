from rest_framework import serializers

from qwen import normalize_landmark_payload


class LandmarkGuideRequestSerializer(serializers.Serializer):
    """The JSON returned by landmark recognition, or the Vision annotation."""

    def to_internal_value(self, data):
        if not isinstance(data, dict):
            raise serializers.ValidationError(
                "Expected a landmark recognition object."
            )
        try:
            return normalize_landmark_payload(data)
        except (TypeError, ValueError):
            raise serializers.ValidationError(
                "Landmark recognition data could not be read."
            )

    def validate(self, attrs):
        landmark = attrs.get("landmark", "")
        if not landmark:
            raise serializers.ValidationError(
                {"landmark": "A landmark name is required."}
            )
        if landmark.lower() == "unknown":
            raise serializers.ValidationError(
                {
                    "landmark": (
                        "No landmark was identified, so there is nothing to describe."
                    )
                }
            )
        return attrs
