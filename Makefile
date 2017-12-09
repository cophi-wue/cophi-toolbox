init :
	pip install --upgrade -r requirements-dev.txt

test : 
	nosetests
