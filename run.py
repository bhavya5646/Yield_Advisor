"""python run.py  -> analyse ./data, then open the dashboard in your browser."""
import subprocess, sys, webbrowser, os
subprocess.check_call([sys.executable, 'analyze.py'])
webbrowser.open('file://' + os.path.abspath('dashboard.html'))
