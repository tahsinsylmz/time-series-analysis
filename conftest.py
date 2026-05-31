"""pytest icin proje kokunu sys.path'e ekler (src paketini import edebilmek icin)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
