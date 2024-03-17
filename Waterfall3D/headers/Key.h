#ifndef KEY_H
#define KEY_H

class Key
{
private:
	bool down;
	bool up;
	bool pressed;
	bool released;

public:
	Key();

	void setDown();
	void setUp();
	bool getHeld();
	bool getPressed();
	bool getReleased();
	void updateStates();
};

#endif // !KEY_H