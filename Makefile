default: main

main: main.cpp kernel.cl
	g++ $< -lOpenCL -Wall -o $@

clean:
	@rm main
