import sys
import argparse
import socket
import driver2 as driver 

def main():
    parser = argparse.ArgumentParser(
        description='Python client to connect to the TORCS SCRC server.'
    )
    parser.add_argument(
        '--host', dest='host_ip', default='localhost',
        help='Host IP address (default: localhost)'
    )
    parser.add_argument(
        '--port', dest='host_port', type=int, default=3001,
        help='Host port number (default: 3001)'
    )
    parser.add_argument(
        '--id', dest='bot_id', default='SCR',
        help='Bot ID (default: SCR)'
    )
    parser.add_argument(
        '--maxEpisodes', dest='max_episodes', type=int, default=1,
        help='Maximum number of learning episodes (default: 1)'
    )
    parser.add_argument(
        '--maxSteps', dest='max_steps', type=int, default=0,
        help='Maximum number of steps (default: 0)'
    )
    parser.add_argument(
        '--track', dest='track', default=None,
        help='Name of the track'
    )
    parser.add_argument(
        '--stage', dest='stage', type=int, default=3,
        help='Stage (0 - Warm-Up, 1 - Qualifying, 2 - Race, 3 - Unknown)'
    )

    args = parser.parse_args()

    print(f"Connecting to server host ip: {args.host_ip} @ port: {args.host_port}")
    print(f"Bot ID: {args.bot_id}")
    print(f"Maximum episodes: {args.max_episodes}")
    print(f"Maximum steps: {args.max_steps}")
    print(f"Track: {args.track}")
    print(f"Stage: {args.stage}")
    print("*********************************************")

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except socket.error as err:
        print(f"Could not create socket: {err}")
        sys.exit(-1)

    sock.settimeout(1.0)
    shutdown_client = False
    cur_episode = 0
    verbose = False
    d = driver.Driver(args.stage)

    while not shutdown_client:
        # Identification handshake
        while True:
            print(f"Sending id to server: {args.bot_id}")
            init_str = d.init()
            buf_out = args.bot_id + init_str
            print(f"Sending init string to server: {buf_out}")

            try:
                sock.sendto(buf_out.encode('utf-8'), (args.host_ip, args.host_port))
            except socket.error as err:
                print(f"Failed to send data: {err}. Exiting.")
                sys.exit(-1)

            try:
                data, _ = sock.recvfrom(1024)
                buf_in = data.decode('utf-8')
            except socket.timeout:
                print("No response from server, retrying...")
                continue
            except socket.error as err:
                print(f"Receive error: {err}")
                continue

            if '***identified***' in buf_in:
                print(f"Received: {buf_in}")
                break

        current_step = 0

        # Main loop per episode
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                buf_in = data.decode('utf-8')
            except socket.timeout:
                if verbose:
                    print("No data received from server.")
                continue
            except socket.error as err:
                print(f"Receive error: {err}")
                continue

            if verbose:
                print(f"Received: {buf_in}")

            if buf_in and '***shutdown***' in buf_in:
                d.onShutDown()
                shutdown_client = True
                print('Client Shutdown')
                break

            if buf_in and '***restart***' in buf_in:
                d.onRestart()
                print('Client Restart')
                break

            current_step += 1
            if args.max_steps and current_step > args.max_steps:
                buf_out = '(meta 1)'
            else:
                buf_out = d.drive(buf_in)

            if verbose:
                print(f"Sending: {buf_out}")

            if buf_out:
                try:
                    sock.sendto(buf_out.encode('utf-8'), (args.host_ip, args.host_port))
                except socket.error as err:
                    print(f"Failed to send data: {err}. Exiting.")
                    sys.exit(-1)

        cur_episode += 1
        if cur_episode >= args.max_episodes:
            shutdown_client = True

    sock.close()


if __name__ == '__main__':
    main()
