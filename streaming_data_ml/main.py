import faust

class Text(faust.Record):
    text: str

app = faust.App('streaming_data_ml', broker='kafka://localhost:9092')

topic = app.topic('text_topic', value_type=Text)

@app.agent(topic)
async def process_text(texts):
    async for text in texts:
        print(text)

if __name__ == '__main__':
    app.main()
