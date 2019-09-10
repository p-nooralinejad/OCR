import shutil
import os
import sys
import curses

class terminalWriter:
	ROWS=0
	COLS = 0
	
	def __init__(self):
		self.ROWS = os.get_terminal_size()[1]
		self.COLS = os.get_terminal_size()[0]
	
	def get_current_position(self):
		return 0

	def write_progress_bar(self,Str):
		print(Str)
		print("\033[2A")

	def write(self,x,y,Str):
		print("\033[" + str(y) + ";" + str(x) + "H" + Str)

	def clear(self):
		os.system("clear")

	def move_cursor(self,x,y):
		print("\033[%d;%dH" % (y, x))

	def get_rows(self):
		return self.ROWS;

	def get_columns(self):
		return self.COLS;

	def done(self):
		print("")