data_lines = []
with open('scratch.txt') as f:
	lines = f.readlines()
	for line in lines:
		data = [el.strip() for el in line.split(' ') if el != '']
		data_lines.append(data)

with open('table1_latex.txt', 'w') as g:
	s = ''
	for row in data_lines:
		for el in row:
			s += el + ' & '
		s = s[:-3]
		s += ' \\\\'
	g.write(s)

