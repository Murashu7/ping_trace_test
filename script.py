import subprocess
import re
import pandas as pd
import datetime
import os
import asyncio
from aiofiles import open as aio_open  # 非同期でファイルを開く

# IPアドレスを抽出する正規表現
ip_pattern = re.compile(r'(\d+\.\d+\.\d+\.\d+)')

# IPアドレスを抽出して配列に格納
def extract_ips(trace_result):
    ip_addresses = []
    for hop in trace_result:
        ip_match = ip_pattern.search(hop)
        if ip_match:
            ip_addresses.append(ip_match.group(0))
    return ip_addresses

# 共通の要素があるかどうかをチェックする関数
def check_route_match(ip_list, expected_route):
    # リストをsetに変換して共通要素を探す
    return bool(set(ip_list) & set(expected_route))

async def ping(host):
    """pingコマンドを非同期で実行"""
    proc = await asyncio.create_subprocess_exec(
        'ping', '-c', '5', host, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    return stdout.decode()

async def traceroute(host):
    """tracerouteコマンドを非同期で実行"""
    proc = await asyncio.create_subprocess_exec(
        'traceroute', '-n', '-m 10', host, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    stdout, _ = await proc.communicate()
    return stdout.decode()

# TODO:
def validate_ping(output, expected_status, max_rtt=100, max_packet_loss=20):
    """
    pingの出力と期待される結果を比較
    output: pingの出力
    expected_status: pingの結果がokかngか判定
    max_rtt: 100ms以下であることを確認
    max_packet_loss: 20%以下であることを確認
    """
    if output is None:
        return False
    
    if expected_status == 'ng':
        if "Request timeout" in output or "Destination Host Unreachable" in output:
            return True

    # パケットロス率の解析
    packet_loss = re.search(r'(\d+)% packet loss', output)
    if packet_loss:
        loss_percentage = int(packet_loss.group(1))  # パケットロス率を整数として取得
    else:
        print("パケットロス率を解析できませんでした")
        return False

    # RTT (応答時間) の解析
    rtt_match = re.search(r'round-trip min/avg/max/stddev = ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms', output)
    if rtt_match:
        avg_rtt = float(rtt_match.group(2))  # 平均RTTを取得
    else:
        print("RTTデータを解析できませんでした")
        return False

    # パケットロス率とRTTの判定
    if loss_percentage <= max_packet_loss and avg_rtt <= max_rtt:
        print(f"成功 (パケットロス: {loss_percentage}%, 平均RTT: {avg_rtt} ms)")
        return True 
    else:
        print(f"失敗 (パケットロス: {loss_percentage}%, 平均RTT: {avg_rtt} ms)")
        return False

def validate_route(output, expected_route):
    """
    Tracerouteの出力と期待される経路を比較
    output: トレースルートの出力
    expected_route: 経路として期待されるホストリスト
    """
    if output is None:
        return False
    # 各ホップのIPアドレスを抽出
    ip_list = extract_ips(output.splitlines())
    return check_route_match(ip_list, expected_route)

async def ping_multiple_hosts(expected_ping_status, results_dir):
    tasks = []
    for host, expected_status in expected_ping_status.items():
        task = asyncio.create_task(process_ping(host, expected_status, results_dir))
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

async def process_ping(host, expected_status, results_dir):
    """個別のホストのpingを処理"""
    full_output_path = os.path.join(results_dir, f'{host}の結果.log')
    p_current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{host} へのpingを確認中...")
    ping_output = await ping(host)
    print(ping_output)
    
    if ping_output:
        async with aio_open(full_output_path, 'a') as f:
            await f.write(f"{host}へのping結果 {p_current_time}\n\n")
            await f.write(f"{ping_output}")
            await f.write("="*40 + "\n\n")
    
    success = validate_ping(ping_output, expected_status)
    if success:
        print(f"{host} へのpingの結果は成功しました\n")
    else:
        print(f"{host} へのpingの結果は失敗しました\n")
    
    return {host: success}

async def trace_multiple_hosts(hosts_with_expected_routes, results_dir):
    tasks = []
    for host, expected_route in hosts_with_expected_routes.items():
        task = asyncio.create_task(process_trace(host, expected_route, results_dir))
        tasks.append(task)
    
    return await asyncio.gather(*tasks)

async def process_trace(host, expected_route, results_dir):
    """個別のホストのtracerouteを処理"""
    full_output_path = os.path.join(results_dir, f'{host}への結果.log')
    t_current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{host} の経路を確認中...")
    trace_output = await traceroute(host)
    print(trace_output)
    
    if trace_output:
        async with aio_open(full_output_path, 'a') as f:
            await f.write(f"{host}へのtrace結果 {t_current_time}\n\n")
            await f.write(f"{trace_output}")
            await f.write("="*40 + "\n\n")
    
    success = validate_route(trace_output, expected_route)
    if success:
        print(f"{host} は期待される経路を通過しました\n")
    else:
        print(f"{host} は期待される経路を通過しませんでした\n")
    
    return {host: success}

async def ping_trace_multiple_hosts(expected_ping_status, hosts_with_expected_routes, selected_name, selected_test_type):
    ping_results = {}
    trace_results = {}
    
    # ファイルを一階層上の "results/selected_name" フォルダに保存
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', f'results/{selected_name}/{selected_test_type}')
    
    # フォルダが存在しなければ作成
    os.makedirs(results_dir, exist_ok=True)
    
    ping_results = await ping_multiple_hosts(expected_ping_status, results_dir)
    trace_results = await trace_multiple_hosts(hosts_with_expected_routes, results_dir)
    return ping_results, trace_results

# 結果を辞書に変換する関数
def convert_to_dict(df):
    ping_list = {}
    trace_list = {}
    
    # 各行を辞書に格納
    for index, row in df.iterrows():
        dest = row['dest']
        ping_list[dest] = row['ping_eval']
        trace_list[dest] = [row['trace_eval_1'], row['trace_eval_2'], row['trace_eval_3']]
        
    return ping_list, trace_list

def select_kyoten(df):
    # 画面に全てのnumberとnameを表示
    print("全てのnumberとnameを表示します:")
    for index, row in df.iterrows():
        print(f"拠点番号: {row['number']}, エリア: {row['area']}, 拠点名: {row['name']}")

    # ユーザーが正しいnumberを選択するまでループ
    selected_row = None
    while selected_row is None or selected_row.empty:
        try:
            # numberを選択するメッセージを表示
            selected_number = int(input("試験をする拠点番号を選択してください: "))
            
            # 選択されたnumberに対応する行を取得
            selected_row = df[df['number'] == selected_number]
            
            if selected_row.empty:
                print("選択したnumberは存在しません。もう一度選択してください。")
        except ValueError:
            print("無効な入力です。数字を入力してください。")

    # 正しいnumberが選択されたらnameとtypeを変数に格納
    selected_area = selected_row['area'].values[0]
    selected_name = selected_row['name'].values[0]
    selected_kyoten_type = selected_row['type'].values[0]
    print(f"選択された拠点: {selected_name}({selected_kyoten_type})")
    return selected_name, selected_kyoten_type

def select_test_type(selected_kyoten_type):
    # 画面に全ての試験内容を表示
    print("試験内容を表示します:")
    if selected_kyoten_type == '大規模':
        numbers = [1, 2, 3, 4, 5]
        types = ['正常性試験・復旧試験・L2SW試験', '帯域保証網試験', 'ベストエフォート網試験', 'ルータ1号機', 'ルータ2号機']
        for index, row in enumerate(types):
            print(f"番号: {index+1}, {row}")
    elif selected_kyoten_type == '中規模':
        numbers = [1, 2, 3]
        types = ['正常性試験・復旧試験・L2SW試験', '帯域保証網試験', 'ベストエフォート網試験']
        for index, row in enumerate(types):
            print(f"番号: {index+1}, {row}")
    elif selected_kyoten_type == '小規模':
        numbers = [1, 2]
        types = ['正常性試験・復旧試験・L2SW試験', 'ベストエフォート網試験']
        for index, row in enumerate(types):
            print(f"番号: {index+1}, {row}")
    # TODO: その他拠点
    else:
        print("その他拠点は試験ができません。")
        return
    
    data = {
        'number': numbers,
        'type': types
    }
    df = pd.DataFrame(data)
    
    # ユーザーが正しいnumberを選択するまでループ
    selected_row = None
    while selected_row is None or selected_row.empty:
        try:
            # numberを選択するメッセージを表示
            selected_number = int(input("試験番号を選択してください: "))
            
            # 選択されたnumberに対応する行を取得
            selected_row = df[df['number'] == selected_number]
            
            if selected_row.empty:
                print("選択したnumberは存在しません。もう一度選択してください。")
        except ValueError:
            print("無効な入力です。数字を入力してください。")

    # 正しいnumberが選択されたらnameとtypeを変数に格納
    selected_test_type = selected_row['type'].values[0]
    print(f"選択された試験: {selected_test_type}")
    return selected_test_type

# テスト実行
def main():
    # CSVファイルの読み込み
    csv_kyoten_list = '../files/kyoten_list.csv' 
    df_kyoten = pd.read_csv(csv_kyoten_list)
    selected_name, selected_kyoten_type = select_kyoten(df_kyoten)
    selected_test_type = select_test_type(selected_kyoten_type)
    
    # 拠点のタイプから判定表を指定する
    if selected_kyoten_type == "大規模":
        if selected_test_type == "常性試験・復旧試験・L2SW試験":
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
        elif selected_test_type == "帯域保証網試験":
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
        elif selected_test_type == "ベストエフォート網試験":
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
        elif selected_test_type == "ルータ1号機":
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
        else:
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
    elif selected_kyoten_type == "中規模":
        if selected_test_type == "常性試験・復旧試験・L2SW試験":
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
        elif selected_test_type == "帯域保証網試験":
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
        else:
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
    elif selected_kyoten_type == "小規模":
        if selected_test_type == "常性試験・復旧試験・L2SW試験":
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
        else:
            csv_net_test_eval = '../files/network_test_evaluation.csv'  # CSVファイルのパスを指定
    # TODO: その他拠点
    else:
        return
    
    # pandasでCSVを読み込む
    df_csv_net_test_eval = pd.read_csv(csv_net_test_eval)

    # 複数のホストと期待されるpingの結果とtrace経路のリスト
    expected_ping_status, hosts_with_expected_routes = convert_to_dict(df_csv_net_test_eval)
    ping_results, trace_results = asyncio.run(ping_trace_multiple_hosts(expected_ping_status, hosts_with_expected_routes, selected_name, selected_test_type))

    print(ping_results, trace_results)

if __name__ == '__main__':
    main()