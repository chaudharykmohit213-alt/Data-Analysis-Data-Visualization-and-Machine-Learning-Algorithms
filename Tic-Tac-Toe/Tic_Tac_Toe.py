def display_latest(row1,row2,row3):
    print(row1[0] +'|'+ row1[1]+ '|' +row1[2]+"              "+"1|2|3") 
    print("------"+"            "+"------")
    print(row2[0]+ '|'+ row2[1]+ '|'+ row2[2]+"              "+"4|5|6")
    print("------"+"            "+"------")
    print(row3[0] +'|'+ row3[1]+ '|'+ row3[2]+"              "+"7|8|9")

def replace_value(r1=None,r2=None,r3=None,sign='W',pos=100):
    if (pos>0 and pos<4):
        r1[pos-1]=sign
    elif(pos>3 and pos<7):
        r2[pos-4]=sign
    elif(pos>6 and pos<10):
        r3[pos-7]=sign

    return (r1,r2,r3)

def check_if_won(ro1,ro2,ro3):
    y1=[ro1[0],ro2[0],ro3[0]]
    y2=[ro1[1],ro2[1],ro3[1]]
    y3=[ro1[2],ro2[2],ro3[2]]
    z1=[ro1[0],ro2[1],ro3[2]]
    z2=[ro1[2],ro2[1],ro3[0]]
    w1=['X','X','X']
    w2=['O','O','O']
    if(ro1==w1 or ro1==w2):
        return True
    elif(ro2==w1 or ro2==w2):
        return True
    elif(ro3==w1 or ro3==w2):
        return True
    elif(y1==w1 or y1==w2):
        return True
    elif(y2==w1 or y2==w2):
        return True
    elif(y3==w1 or y3==w2):
        return True
    elif(z1==w1 or z1==w2):
        return True
    elif(z2==w1 or z2==w2):
        return True
    else:
        return False

def check_valid(sign,pos):
    global prevchar
    global checklist
    if sign.isalpha():
        if(sign=="X" or sign=="O"):
            if pos not in checklist:
                checklist.append(pos)
                return True
                    
                        
    else:
        
        return False

print("WELCOME TO TIC TAC TOE GAME")
row1 = [' ',' ',' ']
row2 = [' ',' ',' ']
row3 = [' ',' ',' ']

display_latest(row1,row2,row3)
checklist=[]
s='U'
p=100
pnum =1
prevchar ='Q'
while(check_if_won(row1,row2,row3)!=True):
    print('Which sign does player {} want to choose X or O ?'.format(pnum))
    s = input()
    while s not in ['X','O']:
        print("Please enter a valid sign")
        s = input()
    while (s==prevchar):
        print("This is the wrong choice since this was choice of previous turn,please re-enter")
        s = input()
    print('Which position does player {} want to place this at ?'.format(pnum))
    p = input()
    while int(p) not in [1,2,3,4,5,6,7,8,9]:
        print("Sorry please enter a valid number between 0 and 9")
        p=input()
    print("Sign is = "+s+"Prevchar as per program is = "+prevchar)
    condition=check_valid(s,int(p))
    print(condition)
    if condition:
        row1,row2,row3 = replace_value(row1,row2,row3,s,int(p))
        display_latest(row1,row2,row3)
        if pnum==1:
            pnum =2
        else:
            pnum =1
        prevchar=s
    else:
        print("Oops either the sign or the position is wrong.Please re enter position and sign")
        pass
else:
     if pnum==1:
        pnum =2
     else:
        pnum =1
     print("Congratulations player {} has won the game".format(pnum))



