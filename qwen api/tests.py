import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings
from rest_framework.test import APIClient

from qwen import parse_guide
from recognition.services.vision import recognize_landmark


GUIDE = {
    "summary": "The Eiffel Tower is an iron tower on the Champ de Mars in Paris.",
    "why_visit": "It is the viewpoint most visitors use to understand the city's layout.",
    "history": "It was built for the 1889 Exposition Universelle and was kept as a radio mast.",
    "what_to_see": ["The summit", "The iron lattice from underneath"],
    "best_time_to_visit": "Arrive at opening to avoid the longest queue.",
    "how_long_to_stay": "Allow about two hours, more if you summit.",
    "getting_there": "Metro line 6 to Bir-Hakeim, or walk from Trocadero.",
    "tickets_and_entry": "Book a timed entry ticket and confirm the day's hours on arrival.",
    "accessibility": "Elevators reach some levels. Check step-free access before you go.",
    "what_to_bring": ["A light jacket for the summit"],
    "food_and_drink": ["A crepe stand on Rue de l'Universite"],
    "nearby": ["Trocadero gardens", "Seine river cruise departure"],
    "etiquette": ["Keep bags closed in the crowds"],
    "photo_spots": ["Trocadero esplanade"],
    "with_children": "Elevators are easier than the open stairs.",
    "safety": "Watch for pickpockets around the base.",
    "suggested_visit": ["Photograph it from Trocadero", "Go up the tower"],
}


class GuideServer(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        self.server.captured = {"path": self.path, "body": body}
        if self.server.fail:
            payload = json.dumps({"error": "model failed"}).encode()
            status = 500
        else:
            payload = json.dumps(self.server.answer).encode()
            status = 200
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        return


def start_guide_server(answer, fail=False):
    server = ThreadingHTTPServer(("127.0.0.1", 0), GuideServer)
    server.answer = answer
    server.fail = fail
    server.captured = None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class TouristGuideTests(SimpleTestCase):
    def setUp(self):
        self.client = APIClient()

    def test_guide_sends_google_result_to_local_qwen(self):
        server = start_guide_server(
            {"message": {"role": "assistant", "content": json.dumps(GUIDE)}}
        )
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        base_url = "http://127.0.0.1:%s" % server.server_address[1]
        recognition = {
            "success": True,
            "result": {
                "landmark": "Eiffel Tower",
                "confidence": 92.34,
                "locations": [
                    {"latitude": 48.85837, "longitude": 2.294481}
                ],
            },
        }

        with override_settings(
            QWEN_BASE_URL=base_url,
            QWEN_MODEL="qwen2.5",
            QWEN_API="ollama",
            QWEN_TIMEOUT=5,
        ):
            response = self.client.post(
                "/api/recognition/guide/",
                recognition,
                format="json",
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["success"])
        self.assertTrue(body["parsed"])
        self.assertEqual(body["model"], "qwen2.5")
        self.assertEqual(body["source"]["landmark"], "Eiffel Tower")
        self.assertEqual(body["guide"]["summary"], GUIDE["summary"])
        self.assertEqual(body["guide"]["what_to_see"], GUIDE["what_to_see"])
        self.assertEqual(body["guide"]["suggested_visit"], GUIDE["suggested_visit"])

        sent = server.captured
        self.assertEqual(sent["path"], "/api/chat")
        self.assertEqual(sent["body"]["model"], "qwen2.5")
        self.assertFalse(sent["body"]["stream"])
        self.assertEqual(sent["body"]["format"], "json")
        system = sent["body"]["messages"][0]["content"]
        user = sent["body"]["messages"][1]["content"]
        self.assertIn("on-the-spot guide", system)
        self.assertIn("Do not invent ticket prices", system)
        self.assertIn("Eiffel Tower", user)
        self.assertIn("48.85837", user)

    def test_guide_accepts_raw_vision_annotation(self):
        server = start_guide_server(
            {"message": {"role": "assistant", "content": json.dumps(GUIDE)}}
        )
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        base_url = "http://127.0.0.1:%s" % server.server_address[1]
        raw = {
            "landmarkAnnotations": [
                {
                    "description": "Eiffel Tower",
                    "score": 0.9234,
                    "locations": [
                        {
                            "latLng": {
                                "latitude": 48.85837,
                                "longitude": 2.294481,
                            }
                        }
                    ],
                }
            ]
        }

        with override_settings(
            QWEN_BASE_URL=base_url,
            QWEN_MODEL="qwen2.5",
            QWEN_API="ollama",
            QWEN_TIMEOUT=5,
        ):
            response = self.client.post(
                "/api/recognition/guide/",
                raw,
                format="json",
            )

        self.assertEqual(response.status_code, 200)
        source = response.json()["source"]
        self.assertEqual(source["landmark"], "Eiffel Tower")
        self.assertEqual(source["confidence"], 92.34)
        self.assertEqual(
            source["locations"],
            [{"latitude": 48.85837, "longitude": 2.294481}],
        )

    def test_unknown_landmark_does_not_call_qwen(self):
        response = self.client.post(
            "/api/recognition/guide/",
            {"success": True, "result": {"landmark": "Unknown", "confidence": 0}},
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertFalse(response.json()["success"])
        self.assertIn("nothing to describe", response.json()["errors"]["landmark"][0])

    def test_missing_landmark_is_rejected(self):
        response = self.client.post(
            "/api/recognition/guide/",
            {"success": False},
            format="json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("landmark", response.json()["errors"])

    @override_settings(
        QWEN_BASE_URL="http://127.0.0.1:9",
        QWEN_MODEL="qwen2.5",
        QWEN_API="ollama",
        QWEN_TIMEOUT=1,
    )
    def test_unreachable_qwen_returns_503(self):
        response = self.client.post(
            "/api/recognition/guide/",
            {"landmark": "Eiffel Tower", "confidence": 90},
            format="json",
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("not reachable", response.json()["errors"]["qwen"][0])

    def test_qwen_http_error_returns_502(self):
        server = start_guide_server({}, fail=True)
        self.addCleanup(server.shutdown)
        self.addCleanup(server.server_close)
        base_url = "http://127.0.0.1:%s" % server.server_address[1]

        with override_settings(
            QWEN_BASE_URL=base_url,
            QWEN_MODEL="qwen2.5",
            QWEN_API="ollama",
            QWEN_TIMEOUT=5,
        ):
            response = self.client.post(
                "/api/recognition/guide/",
                {"landmark": "Eiffel Tower", "confidence": 90},
                format="json",
            )

        self.assertEqual(response.status_code, 502)
        self.assertIn("HTTP 500", response.json()["errors"]["qwen"][0])

    def test_fenced_guide_json_is_parsed(self):
        content = "```json\n%s\n```" % json.dumps(
            {"summary": "A tower.", "what_to_see": "The summit"}
        )

        guide, parsed = parse_guide(content)

        self.assertTrue(parsed)
        self.assertEqual(guide["summary"], "A tower.")
        self.assertEqual(guide["what_to_see"], ["The summit"])
        self.assertEqual(guide["nearby"], [])

    def test_plain_text_guide_is_still_returned(self):
        guide, parsed = parse_guide("Visit the tower early.")

        self.assertFalse(parsed)
        self.assertEqual(guide["summary"], "Visit the tower early.")
        self.assertEqual(guide["suggested_visit"], [])


class RecognizeLandmarkTests(SimpleTestCase):
    def test_google_location_is_included_for_the_guide(self):
        landmark = SimpleNamespace(
            description="Eiffel Tower",
            score=0.91,
            locations=[
                SimpleNamespace(
                    lat_lng=SimpleNamespace(latitude=48.85837, longitude=2.294481)
                )
            ],
        )
        vision_response = SimpleNamespace(landmark_annotations=[landmark])

        class FakeClient:
            def landmark_detection(self, image):
                self.content = image.content
                return vision_response

        fake = FakeClient()
        with patch(
            "recognition.services.vision.vision.ImageAnnotatorClient",
            return_value=fake,
        ):
            result = recognize_landmark(BytesIO(b"image-bytes"))

        self.assertEqual(fake.content, b"image-bytes")
        self.assertEqual(result["landmark"], "Eiffel Tower")
        self.assertEqual(result["confidence"], 91.0)
        self.assertEqual(
            result["locations"],
            [{"latitude": 48.85837, "longitude": 2.294481}],
        )
