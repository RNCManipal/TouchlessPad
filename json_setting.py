import json

settings = {}
settings = json.loads(open("settings.json").read())

settings['N'] = "Open Notepad"
settings['K'] = 'Keyboard'

# make empty settings file
with open('settings.json', 'w') as f:
    json.dump(settings, f)