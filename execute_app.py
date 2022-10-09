from json_setting import settings
import os
import webbrowser
import keyboard


def execute(labels):  
    funct = ""

    for val in settings.keys():
        if val == labels:
            funct = settings[val]
    
    if (funct == "Amazon"):
        webbrowser.open_new_tab("https://www.amazon.in")
    
    if (funct == "Bing"):
        webbrowser.open_new_tab("https://www.bing.com/")

    if (funct == "Calculator"):
        os.startfile(r"C:\Windows\System32\Calc.exe")

    if (funct == "Discord"):
        webbrowser.open_new_tab("https://discord.com/login")
        
    if (funct == "Ebay"):
        webbrowser.open_new_tab("https://www.ebay.com")
    
    if (funct == "Facebook"):
        webbrowser.open_new_tab("https://www.facebook.com")

    if (funct == "Google"):
        webbrowser.open_new_tab("www.google.com")

    if (funct == "Hulu"):
        webbrowser.open_new_tab("https://www.hulu.com")
        
    if (funct == "Instagram"):
       webbrowser.open_new_tab("https://www.instagram.com")
    
    if (funct == "JP"):
        webbrowser.open_new_tab("https://www.jpmorganchase.com/")
    
    #if (funct == "Keyboard"):
    #    os.startfile(r"C:\Program Files\Common Files\microsoft shared\ink\TabTip.exe")

    if (funct == "MIT"):
        webbrowser.open_new_tab("https://manipal.edu/mit.html")
    
    if (funct == "Notepad"):
        os.startfile(r"C:\Windows\System32\Notepad.exe")
    
    if (funct == "Outlook"):
        webbrowser.open_new_tab("https://outlook.live.com/mail/0/")

    if (funct == "This PC"):
        keyboard.send('win + e')

    if (funct == "Quora"):
        webbrowser.open_new_tab("https://www.quora.com")
    
    if (funct == "Reddit"):
        webbrowser.open_new_tab("https://www.reddit.com")
    
    if (funct == "Spotify"):
        webbrowser.open_new_tab("https://open.spotify.com")

    if (funct == "Twitter"):
        webbrowser.open_new_tab("https://twitter.com")

    if (funct == "Unicef"):
        webbrowser.open_new_tab("https://www.unicef.org/")
        
    if funct == "Valve":
        webbrowser.open_new_tab("https://store.steampowered.com/")
    
    if (funct == "Whatsapp"):
        webbrowser.open_new_tab("https://web.whatsapp.com/")

    if (funct == "Exit"):
        keyboard.send("esc")

    if (funct == "YouTube"):
        webbrowser.open_new_tab("https://www.youtube.com")
    
    if (funct == "Zoom"):
        webbrowser.open_new_tab("https://zoom.us")

