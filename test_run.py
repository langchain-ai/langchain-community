# We try to import it from the top-level package
# If this works, it means your __init__.py edit was successful
try:
    from langchain_community.document_loaders import GeniusLoader

    print("✅ STEP 1 PASSED: GeniusLoader was imported successfully!")
except ImportError as e:
    print("❌ STEP 1 FAILED: Could not import GeniusLoader.")
    print(e)
    exit()

# Now we try to use it
# Note: If you don't have a real token, we expect an error, but NOT a crash.
try:
    print("attempting to initialize loader...")
    # Replace 'fake_token' with a real one if you want actual lyrics
    loader = GeniusLoader("Taylor Swift", api_token="fake_token")
    print("✅ STEP 2 PASSED: Loader initialized!")

    print("Attempting to load data...")
    docs = list(loader.lazy_load())

    if docs:
        print(f"✅ STEP 3 PASSED: Found song: {docs[0].metadata['title']}")
    else:
        print("⚠️ STEP 3: No docs found (Expected if token is fake).")

except Exception as e:
    # If it fails because of the API token, that is actually GOOD.
    # It means your code ran and tried to hit the API.
    if "401" in str(e) or "403" in str(e) or "Token" in str(e):
        print(
            "✅ STEP 3 PASSED: Code ran! (API rejected the fake token, which is expected)."
        )
    else:
        print(f"❌ STEP 3 FAILED with unexpected error: {e}")
