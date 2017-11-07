'''

question1: 
Design the data structures for a generic deck of cards. Explain how you would subclass
it to implement particular card games

two classes:

deck class
	constructor(playing_deck, num of decks)
	attributes: num of decks, playing_deck arrlist of arr, used_deck arrlist of arr
	methods:
		shuffle deck(input deck)
		deal(arr of deck, num of people to deal)
		draw_card(deck)
		peek(deck)
		sort deck(deck)
		reset()--> shuffle and put all into playing_deck
		make_used --> put into used_deck
		next_used --> shuffle used at front and putinto playing_deck


card interface
	

standard card class
	attributes: suit, number, value, setters and getters

deck maker
	make deck of generic cards suits and stuff


'''