from django.urls import path

from views import TouristGuideView

urlpatterns = [
    path(
        "guide/",
        TouristGuideView.as_view(),
        name="tourist-guide",
    ),
]
