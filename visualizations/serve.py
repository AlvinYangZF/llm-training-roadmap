#!/usr/bin/env python3
"""Serve LLM Algorithm Visualization site on port 8080."""
import http.server
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
handler = http.server.SimpleHTTPRequestHandler
server = http.server.HTTPServer(('0.0.0.0', 8080), handler)
print('Serving LLM Algorithm Visualizations at http://localhost:8080')
print('Press Ctrl+C to stop.')
server.serve_forever()
