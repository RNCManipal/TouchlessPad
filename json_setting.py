import json

settings = {}
settings = json.loads(open("settings.json").read())

settings['A'] = "Amazon"
settings['B'] = "Bing"
settings['C'] = "Calculator"
settings['D'] = "Discord"
settings['E'] = "Ebay"
settings['F'] = "Facebook"
settings['G'] = "Google"
settings['H'] = "Hulu"
settings['I'] = "Instagram"
settings['J'] = "Jupiter Notebook"
settings['K'] = 'Keyboard'
settings['L'] = "Instagram"
settings['M'] = "MIT"
settings['N'] = "Notepad"
settings['O'] = "Outlook"
settings['P'] = "This PC"
settings['Q'] = "Quora"
settings['R'] = "Reddit"
settings['S'] = "Spotify"
settings['T'] = "Twitter"
settings['U'] = "Uber"
settings['V'] = "Valve"
settings['W'] = "Whatsapp"
settings['X'] = "Xfinity"
settings['Y'] = "YouTube"
settings['Z'] = "Zoom"
settings['1'] = "Instagram"
settings['2'] = "2"
settings['3'] = "3"
settings['4'] = "Hulu"
settings['5'] = "Spotify"
settings['6'] = "Google"
settings['7'] = "7"
settings['8'] = "8"
settings['9'] = "9"
settings['0'] = "Outlook"

with open("settings.json", "w") as file:
    json.dump(settings, file, indent=4)
