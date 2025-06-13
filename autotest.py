import asyncio
import pandas as pd
from playwright.async_api import async_playwright

# === Step 1: Load public Google Sheet ===
def load_questions_from_public_sheet():
    url = "https://docs.google.com/spreadsheets/d/14uwfo0KXFhu0A1AU_Afh-GhyC_z5oYVN2t0I2EMnuOw/export?format=csv"
    # url = "https://docs.google.com/spreadsheets/d/1bEFOBPs3Gsdm1TudlrRjyZhx7r1xWbdpB1WyLQdLe3M/export?format=csv"
    df = pd.read_csv(url)
    return df["Questions"].dropna().tolist()

# === Step 2: Ask questions on chatbot site and display answers ===
async def ask_questions_on_chatbot(questions):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()
        await page.goto("https://chatbot-e2fd.onrender.com/")

        for q in questions:
            print(f"\n=== Question: {q} ===")
            await page.wait_for_selector('textarea')
            await page.fill('textarea', q)
            await page.keyboard.press('Enter')

            # Wait for response
            try:
                await page.wait_for_selector(".stChatMessage", timeout=30000)
                await asyncio.sleep(60)  # Wait a bit for the full message to render

                # Get all chat messages and assume the last one is the response
                messages = await page.locator(".stChatMessage").all_text_contents()
                if len(messages) >= 2:
                    response = messages[-1]
                    print(f"Response: {response}")
                else:
                    print("No response detected.")
            except Exception as e:
                print(f"Timeout or error waiting for response to: {q} -> {e}")

        # await browser.close()

# === Run ===
if __name__ == "__main__":
    questions = load_questions_from_public_sheet()
    asyncio.run(ask_questions_on_chatbot(questions))
