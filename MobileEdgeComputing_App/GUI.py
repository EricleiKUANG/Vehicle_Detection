from tkinter import *
import paramiko
from PIL import Image, ImageTk
#top = Tk()
#top.title("term project")
#top.geometry("800x600")

def printhello():
    t.insert(END,"hello\n")
'''
def ssh_scp_put(ip,port,user,password,local_file,remote_file):
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.connect(ip, 22, 'pi', password)
	a = ssh.exec_command('date')
	stdin, stdout, stderr = a
	print(stdout.read())
	sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
	sftp = ssh.open_sftp()
	sftp.put(local_file, remote_file)
'''
def ssh_scp_get(ip, port, user, password, remote_file, local_file, method):
	ssh = paramiko.SSHClient()
	ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
	ssh.connect(ip, port, user, password)
	if method == 1:
		#stdin, stdout, stderr = ssh.exec_command('cd /home/pi/tensorflow/models/research/object_detection;sudo su; python3 ./object_detection_writeFRCNNRes50.py;pwd')
		stdin, stdout, stderr=ssh.exec_command('cd /home/pi/tensorflow/models/research/object_detection;sudo su; sh start.sh ;pwd')
		print('shit')
		
	sftp = paramiko.SFTPClient.from_transport(ssh.get_transport())
	sftp = ssh.open_sftp()
	sftp.get(remote_file, local_file)










ip='10.128.243.168'
#ip='192.168.50.30'
password = '123123'
port = '22'
user = 'pi'
remote_file = '/home/pi/Desktop/result.txt'
local_file = 'C://Users//kuang//Desktop//Project//code//result.txt'
#initiallize
ssh_scp_get(ip,port,user,password,remote_file,local_file,1)



#num=0
 
tk=Tk()
tk.geometry("1300x1000")
tk.title("term project")

canvas=Canvas(tk,width=1200,height=100)

canvas2 = Canvas(tk,width=1280,height=720)
canvas2.pack(side = "bottom")
canvas.pack()


itext=canvas.create_text(200,30,text='connecting, wait for 30 seconds')
#itext2=canvas2.create_text(200,50,text='connecting, wait for 30 seconds')
#tk.after(30000)
while(True):
    ssh_scp_get(ip,port,user,password,remote_file,local_file,2)
    
    f=open('C://Users//kuang//Desktop//Project//code//result.txt','r')
    line = f.readline()
    #num +=1
    l = line.split(' ')
    print(l[0])
    canvas.itemconfig(itext,text=l[0]+' car '+str(l[1])+' truck '+str(l[2])+' bus '+str(l[3]))
    f.close()
    image = Image.open("C://Users//kuang//Desktop//Project//code//outputimg1//outputimg//"+l[0])
    im = ImageTk.PhotoImage(image)
    canvas2.create_image(640,260,image = im)
    canvas.itemconfig(itext,text=l[0]+' car '+str(l[1])+' truck '+str(l[2])+' bus '+str(l[3]))

    canvas.insert(itext,24,'')
    tk.update()
    #print('num=%d'%num)
    tk.after(5000)


'''
t = Text()
t.pack()  


Button(top, text="press", command=printhello).pack()
top.mainloop() 
'''