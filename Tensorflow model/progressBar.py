from terminalWriter import terminalWriter
from decimal import Decimal

class progressBar:
	rows = 0;
	cols = 0;
	total_items= 0;
	items_done = 0;
	title = ""
	tw = terminalWriter()

	def __init__(self, tot_items = 1, il = 0, name = ""):
		self.total_items = tot_items;
		self.items_done = il;
		self.title = name;
		self.rows = self.tw.get_rows();
		self.cols = self.tw.get_columns();
		self.show()

	def show(self):
		
		display = self.title;
		if self.title != "":
			display += ":["
		else:
			display += "["
		# <TITLE>:[
		end = "]"
		end += str(round(100 * float(self.items_done) / float(self.total_items),2))
		end += "%"
		#]<some-number>%
		total_area = self.cols - (len(display) + len(end))

		filled_area = (float(self.items_done) / float(self.total_items))* total_area
		filled_area = int(filled_area)

		for i in range(filled_area):
			display += "#";
		for i in range(total_area - filled_area):
			display += " "
		display += end


		self.tw.write_progress_bar(display);

	def signal_job_done(self):
		self.items_done += 1;
		self.show()
		if(self.items_done == self.total_items):
			self.tw.done();

	def reset(self, new_tot_items, new_name = ""):
		self.total_items = new_tot_items;
		self.title = new_name;
		self.items_done = 0;
