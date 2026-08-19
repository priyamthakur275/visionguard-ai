// VisionGuard AI reference firmware: line protocol + acknowledgement + failsafe.
// Wire motor driver pins to your board/driver before use; this example is not
// hardware validation and must be adapted and tested with wheels off the ground.
const unsigned long COMMAND_TIMEOUT_MS = 750;
unsigned long lastCommandAt = 0;

void stopMotors() { /* Set every motor-driver enable/direction pin to LOW. */ }
void applyCommand(const String& command) {
  if (command == "STOP") stopMotors();
  else if (command == "LEFT") { /* turn left */ }
  else if (command == "RIGHT") { /* turn right */ }
  else if (command == "FORWARD") { /* drive forward */ }
  else if (command == "BACKWARD") { /* drive backward */ }
  else return;
  lastCommandAt = millis();
  Serial.print("ACK "); Serial.println(command);
}
void setup() { Serial.begin(9600); stopMotors(); lastCommandAt = millis(); }
void loop() {
  if (Serial.available()) applyCommand(Serial.readStringUntil('\n'));
  if (millis() - lastCommandAt > COMMAND_TIMEOUT_MS) stopMotors();
}
