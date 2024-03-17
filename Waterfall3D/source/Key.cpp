
#include "Key.h"

Key::Key()
{
	down = false;
	up = true;
	pressed = false;
	released = false;
}

void Key::setDown()
{
	if (up) pressed = true;
	released = false;
	down = true;
	up = false;
}

void Key::setUp()
{
	pressed = false;
	if (down) released = true;
	down = false;
	up = true;
}

bool Key::getHeld()
{
	return down;
}

bool Key::getPressed()
{
	return pressed;
}

bool Key::getReleased()
{
	return released;
}

void Key::updateStates()
{
	pressed = false;
	released = false;
}


