from bootstrap import create_app
# KLJUČNO: Samo jedna linija koda!
# Bootstrap kreira CIJELU aplikaciju, web layer samo konfiguriše
app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)