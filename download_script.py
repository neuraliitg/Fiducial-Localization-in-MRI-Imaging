import urllib
for i in range(1,188):
	url = "https://storage.googleapis.com/mri-storage/SE5/" + "IM" + str(i)
	urllib.urlretrieve (url, "IM" + str(i))
	print("Download IM"+str(i))


