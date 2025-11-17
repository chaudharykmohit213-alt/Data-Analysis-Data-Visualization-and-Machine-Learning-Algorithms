suits = ('Hearts', 'Diamonds', 'Spades', 'Clubs')
ranks = ('Two', 'Three', 'Four', 'Five', 'Six', 'Seven', 'Eight', 'Nine', 'Ten', 'Jack', 'Queen', 'King', 'Ace')
values = {'Two':2, 'Three':3, 'Four':4, 'Five':5, 'Six':6, 'Seven':7, 'Eight':8, 'Nine':9, 'Ten':10, 'Jack':10,
         'Queen':10, 'King':10, 'Ace':11}
import random

class Card:   
    def __init__(self,suit,rank):

        self.suit=suit
        self.rank=rank
        self.value=values[rank]
    
    def __str__(self):
        return f'{self.rank} of {self.suit}'

class Deck:
    
    def __init__(self):
        self.deck = []  # start with an empty list
        for suit in suits:
            for rank in ranks:
                self.deck.append(Card(suit,rank))
    
    def __str__(self):
        tempstr=''
        for card in self.deck:
             tempstr = tempstr+(card.rank+' of '+card.suit+'\n')
        return tempstr
        
    def shuffle(self):
        random.shuffle(self.deck)
        
    def deal(self):
        single_card=self.deck.pop(0)
        return single_card

class Display:
        
       def __init__(self,hand=None,player='',stay=None):

           self.hand=hand
           self.player=player
           self.stay=stay
           self.sum=0
           for card in hand:
               self.sum=self.sum+card.value

       def __str__(self):
           if self.player=="Dealer":
               if self.stay==True:
                   mydis="\nDealer\n#### of ##### ,"
                   for cards in self.hand[1:]:
                       mydis=mydis+cards.rank+' of '+cards.suit+' , '
                   return mydis
               else:
                   mydis="\nDealer\n"
                   for cards in self.hand:
                       mydis=mydis+cards.rank+' of '+cards.suit+' , '
                   mydis = mydis+"  Value = {}".format(self.sum)
                   return mydis
           else:
                mydis='\n'+self.player+'\n'
                for cards in self.hand:
                       mydis=mydis+cards.rank+' of '+cards.suit+' , '
                mydis = mydis+"  Value = {}".format(self.sum)
                return mydis

class Player:

    def __init__(self,name,currenthand=[]):

        self.name=name
        self.currenthand=currenthand

    def __str__(self):

        return "The Player name is :" + self.name

dealerhand=[]
Dealer=Player('Dealer',dealerhand)
playerhand=[]
player_name=input('Please enter the player name : ')
Game_Player=Player(player_name,playerhand)
option = True


while option:

    gamedeck=Deck()
    gamedeck.shuffle()
    
    print(Dealer)
    print(Game_Player)

    dealerhand.clear()    
    dealerhand.append(gamedeck.deal())
    dealerhand.append(gamedeck.deal())

    playerhand.clear()
    playerhand.append(gamedeck.deal())
    playerhand.append(gamedeck.deal())

    dealerdisplay=Display(dealerhand,'Dealer',True)
    playerdisplay=Display(playerhand,player_name,True)

    print(dealerdisplay)
    print(playerdisplay)
    print('\n')
        
    gameon=True
    hitorstay='Hit'
    hidecard=True
    hos=' '

    while gameon:

        print("Do you want to hit or stay :")
        hos=input()
        playersum=0
        dealersum=0
        if hos=='HIT':
            playerhand.append(gamedeck.deal())
            dealerdisplay=Display(dealerhand,'Dealer',hidecard)
            playerdisplay=Display(playerhand,player_name,hidecard)

            print(dealerdisplay)
            print(playerdisplay)

            for pieces in playerhand:
                playersum=playersum+pieces.value

            if playersum>21:
                print("\nSorry you have busted and the Dealer won")
                break        
            
        elif hos=='STAY':
            hidecard=False
            dealerhand.append(gamedeck.deal())
            dealerdisplay=Display(dealerhand,'Dealer',hidecard)
            playerdisplay=Display(playerhand,player_name,hidecard)
            while True:
                dealersum=0
                playersum=0
                
                for pieces in dealerhand:
                    dealersum=dealersum+pieces.value

                for pieces in playerhand:
                    playersum=playersum+pieces.value
                
                if dealersum<17:
                    if playersum<dealersum:
                        dealerdisplay=Display(dealerhand,'Dealer',hidecard)
                        playerdisplay=Display(playerhand,player_name,hidecard)
                        print(dealerdisplay)
                        print(playerdisplay)
                        print("\nThe player {} has lost and the Dealer has won.".format(player_name))
                        break
                    dealerdisplay=Display(dealerhand,'Dealer',hidecard)
                    playerdisplay=Display(playerhand,player_name,hidecard)
                    print(dealerdisplay)
                    print(playerdisplay)
                    dealerhand.append(gamedeck.deal())                   
                elif dealersum>=17 and dealersum<=21:
                    dealerdisplay=Display(dealerhand,'Dealer',hidecard)
                    playerdisplay=Display(playerhand,player_name,hidecard)
                    print(dealerdisplay)
                    print(playerdisplay)
                    print('\n')
                    if playersum>dealersum:
                        print("\nThe player {} has won. Congaratulations!".format(player_name))
                        break
                    elif playersum<dealersum:
                        print("\nThe Dealer has won")
                        break
                    else:
                        print("\nThere is a tie")
                        break
                else:
                    dealerdisplay=Display(dealerhand,'Dealer',hidecard)
                    playerdisplay=Display(playerhand,player_name,hidecard)
                    print(dealerdisplay)
                    print(playerdisplay)
                    print('\n')
                    print("\nDealer busted and the player {} has won!".format(player_name))
                    break
        if hos=='STAY':
            break    
    print("Do you want to play again?")
    option=input()
    if option=='YES':
        option=True
    else:
        option=False


            
            
            
            
        
        
        

    



        

        


    
                
                
                    
                       
                   
                   
            
    

    

    

