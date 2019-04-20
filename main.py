from __future__ import unicode_literals

from hazm import *

normalizer = Normalizer()
normalizer.normalize('اصلاح نويسه ها و استفاده از نیم‌فاصله پردازش را آسان مي كند')

sent_tokenize('ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟')

word_tokenize('ولی برای پردازش، جدا بهتر نیست؟')
tagger = POSTagger(model='resources/postagger.model')
s = tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم'))
print(s)
