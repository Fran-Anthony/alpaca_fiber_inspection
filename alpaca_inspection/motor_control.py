"""Control Arduino/L298N y sus acciones en la interfaz."""
import glob
import os
import time

from PyQt5.QtWidgets import QMessageBox

# PySerial es necesario solo para el control Arduino/L298N.
# Si no está instalado, la parte de visión puede seguir iniciando.
try:
    import serial
    SERIAL_IMPORT_ERROR = None
except Exception as e:
    serial = None
    SERIAL_IMPORT_ERROR = e


class MotorControls:
    def _find_motor_ports(self):
        self.motor_port_combo.clear()

        if serial is None:
            self.motor_port_combo.addItem('PySerial no disponible')
            return

        ports = sorted(glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*'))
        if os.name == 'nt':
            try:
                from serial.tools import list_ports
                ports = [p.device for p in list_ports.comports()]
            except Exception:
                ports = []

        if ports:
            self.motor_port_combo.addItems(ports)
        else:
            self.motor_port_combo.addItem('No encontrado')

    def _toggle_motor_connection(self):
        if serial is None:
            detail = str(SERIAL_IMPORT_ERROR) if SERIAL_IMPORT_ERROR else 'PySerial no está instalado.'
            QMessageBox.warning(
                self,
                'Arduino',
                f'No se puede abrir la conexión serial.\n{detail}\n\n'
                'La detección por visión puede seguir funcionando normalmente.'
            )
            return

        if self.serial_port is not None:
            self._disconnect_motor_serial()
            return

        port = self.motor_port_combo.currentText()
        if not port or port == 'No encontrado':
            QMessageBox.warning(self, 'Arduino', 'No se encontró un puerto Arduino.')
            return

        try:
            self.serial_port = serial.Serial(port, 115200, timeout=0.05)
            time.sleep(1.8)
            self.serial_port.reset_input_buffer()
            self.motor_connect_button.setText('Desconectar')
            self.motor_connection_label.setText(f'Arduino: conectado ({port})')
            self._send_motor_command('STOP')
            self.log(f'Arduino conectado en {port} a 115200 baud.')
        except Exception as e:
            self.serial_port = None
            QMessageBox.critical(self, 'Error de conexión Arduino', str(e))

    def _disconnect_motor_serial(self):
        try:
            if self.serial_port is not None:
                self._send_motor_command('STOP')
                self.serial_port.close()
        except Exception:
            pass

        self.serial_port = None
        self.motor_running = False
        self.motor_connect_button.setText('Conectar')
        self.motor_connection_label.setText('Arduino: desconectado')
        self.log('Arduino desconectado.')

    def _send_motor_command(self, command):
        if self.serial_port is None or not self.serial_port.is_open:
            return False

        try:
            self.serial_port.write((command + '\n').encode('utf-8'))
            self.serial_port.flush()
            return True
        except Exception as e:
            self.log(f'Error serial Arduino: {e}')
            return False

    def _motor_start(self):
        if self.serial_port is None or not self.serial_port.is_open:
            QMessageBox.warning(self, 'Arduino', 'Primero conecta el Arduino.')
            return

        self._send_motor_command(f'SPEED:{self.motor_speed}')
        self._send_motor_command(self.motor_direction)
        self._send_motor_command('START')
        self.motor_running = True
        self.log(f'Motor iniciado. Dirección={self.motor_direction}, PWM={self.motor_speed}.')

    def _motor_stop(self):
        self._send_motor_command('STOP')
        self.motor_running = False
        self.log('Motor detenido.')

    def _motor_forward(self):
        self.motor_direction = 'FORWARD'
        self._send_motor_command('FORWARD')
        self.log('Dirección del motor: ADELANTE.')

    def _motor_reverse(self):
        self.motor_direction = 'REVERSE'
        self._send_motor_command('REVERSE')
        self.log('Dirección del motor: REVERSA.')

    def _motor_speed_changed(self, value):
        self.motor_speed = int(value)
        pct = int(round(self.motor_speed * 100 / 255))
        self.motor_speed_label.setText(f'PWM: {self.motor_speed}  |  {pct} %')

        if self.motor_running:
            self._send_motor_command(f'SPEED:{self.motor_speed}')

    def _read_motor_serial(self):
        if self.serial_port is None or not self.serial_port.is_open:
            return

        try:
            while self.serial_port.in_waiting:
                line = self.serial_port.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    self.motor_response_label.setText(f'Respuesta Arduino: {line}')
        except Exception as e:
            self.log(f'Error leyendo Arduino: {e}')
