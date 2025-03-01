# metinlerde bulunan fazla boslukları ortadan kaldırır
text = "Hello,       World!       2035"

text.split()


cleaned_text1 = " ".join(text.split())
print(f"text: {text} \n cleaned_text1: {cleaned_text1}")

#buyuk -> kucuk harfe cevirme
text ="Hello, World! 2035"

text.lower()

cleaned_text2 = text.lower()
print(f"text: {text} \n cleaned_text2: {cleaned_text2}")

#noktalma işaretlerini kaldırma
import string

text = "Hello, World! 2035"
cleaned_text3 = text.translate(str.maketrans("", "", string.punctuation))
print(f"text: {text} \n cleaned_text3: {cleaned_text3}")

#ozel karakterleri kaldirma 
import re 

text = "Hello, World! 2035%#"
cleaned_text4 = re.sub(r"[^a-zA-Z0-9\s]", "", text)
print(f"text: {text} \n cleaned_text4: {cleaned_text4}")    

#yazim hatalarini duzlet
from textblob import TextBlob   #metin analizlerinde kullanılan bir kütühane    

text = "Hellio, Wirld! 2035"
TextBlob(text).correct() #correct: yazim hatalarini duzeltir
print(f"text: {text} \n cleaned_text5: {TextBlob(text).correct()}")

#html yada url temizleme
from bs4 import BeautifulSoup   

html_text = "<p>Hello, World! 2035</p>"
cleaned_text6 =  BeautifulSoup(html_text, "html.parser").get_text()
print(f"html_text: {html_text} \n cleaned_text6: {cleaned_text6}")
