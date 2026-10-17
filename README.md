# Counterpoint
pip install fastapi uvicorn
cd backend && ADMIN_EMAILS=you@example.com uvicorn main:app --reload
Open http://localhost:8000 . Register with an ADMIN_EMAILS address to get the admin page.
Hook points: generate_reply() (Debate Orchestrator) and get_search_config() (Kaisha's engine) in backend/main.py.
