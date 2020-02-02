import requests
from lxml import etree
from bs4 import BeautifulSoup



urls = []
headers = {
    "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:72.0) Gecko/20100101 Firefox/72.0"
}
for id in range(100):
    print(id)
    for i in range(1,50):
        try:
            if i==1:
                new_url = "http://www.shicimingju.com/chaxun/zuozhe/" + str(id) + ".html"
            else:
                new_url = "http://www.shicimingju.com/chaxun/zuozhe/" + str(id) + "_" + str(i) + ".html"
            r = requests.get(new_url,headers=headers)
            for j in range(2,40,2):
                html = etree.HTML(r.text)
                index = html.xpath("/html/body/div[4]/div[1]/div[1]/div[" + str(j) + "]/div[2]/h3/a/@href")
                urls.append(index)
        except:
            pass

poems = ""
for i in urls:
    try:
        url = "http://www.shicimingju.com" + i[0]
        r = requests.get(url,headers=headers)
        soup = BeautifulSoup(r.text)
        poem = soup.find("div",attrs={"class":"item_content"})
        for j in poem.contents:
            if len(j) == 16 and str(j)[7] == '，' and str(j)[0] != '\n' and str(j)[1] != '\n':
                poems = poems + str(j) + "\n"
                print(str(j))
    except:
        pass

with open(r"data\poems_res.txt","a+",encoding="UTF-8") as f:
    f.write(poems)