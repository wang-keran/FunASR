from funasr import AutoModel

model = AutoModel(model="ct-punc")

res = model.generate(input="那今天的会就到这里吧 happy new year 明年见")
print(res)