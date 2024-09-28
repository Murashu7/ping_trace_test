import pytest
from unittest.mock import patch, AsyncMock
import os
import subprocess
import pandas as pd
from unittest import mock

# テスト対象の関数が記載されているモジュールをインポートします
from script import save_results, extract_ips, check_route_match, validate_ping, validate_route, ping, ping_multiple_hosts, \
    traceroute, trace_multiple_hosts, convert_to_list, select_kyoten, select_test_type, get_test_type_path, SelectionError, \
    generate_unique_filename, get_name_by_host

@pytest.mark.asyncio
async def test_save_results_ping(tmpdir):
    # 一時ディレクトリを作成
    results_dir = tmpdir.mkdir("results")

    # テストデータ
    kyoten_name = "拠点A"
    test_type = "テスト"
    index = 1
    host = "8.8.8.8"
    ping_output = "5 packets transmitted, 5 packets received, 0.0% packet loss"
    result_type = "ping"

    # save_results関数を実行
    await save_results(kyoten_name, test_type, index, host, ping_output, str(results_dir), result_type)

    # 期待されるファイルパス
    expected_file_path = os.path.join(str(results_dir), f'{index}_{kyoten_name}_{test_type}の結果.log')

    # ファイルが作成されたか確認
    assert os.path.exists(expected_file_path)

    # ファイルの内容を確認
    with open(expected_file_path, 'r') as f:
        content = f.read()
        assert f"{host} への {result_type} 結果" in content
        assert ping_output in content
        assert "="*40 in content
  
@pytest.mark.asyncio
async def test_save_results_traceroute(tmpdir):
    # 一時ディレクトリを作成
    results_dir = tmpdir.mkdir("results")

    # テストデータ
    kyoten_name = "拠点A"
    test_type = "テスト"
    index = 1
    host = "8.8.8.8"
    trace_output = "traceroute to 8.8.8.8, 30 hops max"
    result_type = "traceroute"

    # save_results関数を実行
    await save_results(kyoten_name, test_type, index, host, trace_output, str(results_dir), result_type)

    # 期待されるファイルパス
    expected_file_path = os.path.join(str(results_dir), f'{index}_{kyoten_name}_{test_type}の結果.log')

    # ファイルが作成されたか確認
    assert os.path.exists(expected_file_path)

    # ファイルの内容を確認
    with open(expected_file_path, 'r') as f:
        content = f.read()
        assert f"{host} への {result_type} 結果" in content
        assert trace_output in content
        assert "="*40 in content

@pytest.mark.asyncio
async def test_ping_async_valid_host():
    host = 'google.com'  # 有効なホスト
    result = await ping(ping_count='2', host=host)
    assert 'bytes from' in result or 'icmp_seq' in result, "Ping failed for valid host"

@pytest.mark.asyncio
async def test_ping_async_invalid_host():
    host = 'invalid-host'  # 存在しないホスト
    result = await ping(ping_count='2', host=host)
    assert 'ping: cannot resolve' in result or 'unknown host' in result, "Ping should fail for invalid host"

@pytest.mark.asyncio
async def test_ping_async_custom_ping_count():
    host = 'google.com'
    ping_count = '3'
    result = await ping(ping_count=ping_count, host=host)
    
    # 'icmp_seq' が ping_count 分含まれることを確認
    assert result.count('icmp_seq') == int(ping_count), f"Ping count mismatch, expected {ping_count}"

@pytest.fixture
def mock_config_file(tmp_path):
    config_content = 'max_rtt: 100\nmax_packet_loss: 20\n'
    config_file = tmp_path / 'config.txt'
    config_file.write_text(config_content)
    return config_file

def test_validate_ping_success(mock_config_file):
    output = """64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.032 ms
64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.045 ms
64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.034 ms

--- 192.168.1.1 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2001ms
round-trip min/avg/max/stddev = 0.032/0.037/0.045/0.001 ms
"""
    result = validate_ping(max_rtt=100, max_packet_loss=20, output=output, expected_status='ok')
    assert result is True

def test_validate_ping_failure_due_to_rtt(mock_config_file):
    output = """64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=150.032 ms
64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=200.045 ms

--- 192.168.1.1 ping statistics ---
2 packets transmitted, 2 received, 0% packet loss, time 2001ms
round-trip min/avg/max/stddev = 150.032/175.037/200.045/0.001 ms
"""
    result = validate_ping(max_rtt=100, max_packet_loss=20, output=output, expected_status='ok')
    assert result is False

def test_validate_ping_failure_due_to_packet_loss(mock_config_file):
    output = """64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.032 ms
64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.045 ms

--- 192.168.1.1 ping statistics ---
3 packets transmitted, 1 received, 66% packet loss, time 2001ms
round-trip min/avg/max/stddev = 0.032/0.037/0.045/0.001 ms
"""
    result = validate_ping(max_rtt=100, max_packet_loss=20, output=output, expected_status='ok')
    assert result is False

def test_validate_ping_ng_case(mock_config_file):
    output = "Request timeout"
    result = validate_ping(max_rtt=100, max_packet_loss=20, output=output, expected_status='ng')
    assert result is True

def test_validate_ping_ng_but_ok_case(mock_config_file):
    output = """64 bytes from 192.168.1.1: icmp_seq=1 ttl=64 time=0.032 ms
64 bytes from 192.168.1.1: icmp_seq=2 ttl=64 time=0.045 ms
64 bytes from 192.168.1.1: icmp_seq=3 ttl=64 time=0.034 ms

--- 192.168.1.1 ping statistics ---
3 packets transmitted, 3 received, 0% packet loss, time 2001ms
round-trip min/avg/max/stddev = 0.032/0.037/0.045/0.001 ms
"""
    result = validate_ping(max_rtt=100, max_packet_loss=20, output=output, expected_status='ng')
    assert result is False
    
@pytest.mark.asyncio
async def test_ping_multiple_hosts():
    # モックする内容
    mock_process_ping = AsyncMock(side_effect=[
        {"8.8.8.8": True},  # 1回目の呼び出しの戻り値
        {"1.1.1.1": True}   # 2回目の呼び出しの戻り値
    ])
    
    # process_ping をモック
    with patch('script.process_ping', mock_process_ping):
        list_ping_eval = [
            {"8.8.8.8": "ok"},
            {"1.1.1.1": "ng"}
        ]
    
        results = await ping_multiple_hosts(
            ping_count=5,
            max_rtt=100,
            max_packet_loss=20,
            kyoten_name="拠点A",
            test_type="テスト",
            list_ping_eval=list_ping_eval,
            results_dir="/dummy_dir")
        
        # 結果が正しいか確認
        assert results == [{"8.8.8.8": True}, {"1.1.1.1": True}]
        
        # process_pingが2回呼び出されたか確認
        assert mock_process_ping.call_count == 2

# IPアドレス抽出関数のテスト
def test_extract_ips():
    trace_result = [
        "1 192.168.1.1 1.123 ms",
        "2 10.0.0.1 5.456 ms",
        "3 8.8.8.8 2.789 ms"
    ]
    expected_ips = ['192.168.1.1', '10.0.0.1', '8.8.8.8']
    
    assert extract_ips(trace_result) == expected_ips

# ルート一致チェック関数のテスト
def test_check_route_match():
    list_ip = ['192.168.1.1', '10.0.0.1', '8.8.8.8']
    list_expected_route = ['10.0.0.1', '8.8.4.4', '1.1.1.1']
    
    assert check_route_match(list_ip, list_expected_route) == True

    list_expected_route = ['9.9.9.9']
    assert check_route_match(list_ip, list_expected_route) == False

# tracerouteの経路検証関数のテスト
def test_validate_route_success():
    trace_output = """
    traceroute to google.com (8.8.8.8), 30 hops max
    1  192.168.1.1  1.123 ms
    2  10.0.0.1  5.456 ms
    3  8.8.8.8  2.789 ms
    """
    expected_route = ['192.168.1.1', '10.0.0.1', '8.8.8.8']
    
    assert validate_route(trace_output, expected_route) == True

def test_validate_route_failure():
    trace_output = """
    traceroute to google.com (8.8.8.8), 30 hops max
    1  192.168.1.1  1.123 ms
    2  10.0.0.1  5.456 ms
    3  8.8.8.8  2.789 ms
    """
    expected_route = ['9.9.9.9']
    
    assert validate_route(trace_output, expected_route) == False
    
@pytest.mark.asyncio
async def test_traceroute():
    host = "8.8.8.8"
    expected_output = "1  192.168.1.1  1.234 ms\n2  8.8.8.8  2.345 ms\n"

    # subprocessのモック
    with patch('asyncio.create_subprocess_exec') as mock_create_subprocess:
        # モックされたプロセスの戻り値を設定
        mock_proc = AsyncMock()
        mock_create_subprocess.return_value = mock_proc
        mock_proc.communicate.return_value = (expected_output.encode(), b'')

        # traceroute関数を呼び出し
        output = await traceroute(host)

        # 戻り値が期待通りであることを確認
        assert output == expected_output

        # subprocessが正しく呼び出されたか確認
        mock_create_subprocess.assert_called_once_with(
            'traceroute', '-n', '-m 10', host,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # communicateメソッドが呼び出されたか確認
        mock_proc.communicate.assert_called_once()

@pytest.mark.asyncio
async def test_trace_multiple_hosts():
    kyoten_name = "拠点A"
    test_type = "テスト"
    results_dir = "/dummy_dir"

    list_trace_eval = [
        {"8.8.8.8": ["192.168.1.1", "8.8.8.8"]},
        {"1.1.1.1": ["192.168.1.1", "1.1.1.1"]}
    ]
    
    expected_results = [
        {"8.8.8.8": True},  # process_traceから返される結果
        {"1.1.1.1": True}   # process_traceから返される結果
    ]

    # process_traceをモック
    with patch('script.process_trace', new_callable=AsyncMock) as mock_process_trace:
        # モックの戻り値を設定
        mock_process_trace.side_effect = expected_results
        
        # trace_multiple_hosts関数を呼び出す
        results = await trace_multiple_hosts(kyoten_name, test_type, list_trace_eval, results_dir)

        # 結果が期待通りであることを確認
        assert results == expected_results
        
        # process_traceが正しい数だけ呼び出されたか確認
        assert mock_process_trace.call_count == len(list_trace_eval)

        # 呼び出しの内容を確認
        for index, dict_trace_eval in enumerate(list_trace_eval, start=1):
            host = list(dict_trace_eval.keys())[0]
            list_expected_route = dict_trace_eval[host]
            mock_process_trace.assert_any_call(kyoten_name, test_type, index, host, list_expected_route, results_dir)

def test_convert_to_list():
    # テスト用のデータフレームを作成
    data = {
        'dest': ['8.8.8.8', '1.1.1.1'],
        'ping_eval': ['ok', 'timeout'],
        'trace_eval_1': ['192.168.1.1', '192.168.1.1'],
        'trace_eval_2': ['10.0.0.1', '10.0.0.1'],
        'trace_eval_3': ['8.8.8.8', '1.1.1.1']
    }
    df = pd.DataFrame(data)

    # 期待される結果を設定
    expected_ping_eval = [
        {'8.8.8.8': 'ok'},
        {'1.1.1.1': 'timeout'}
    ]
    
    expected_trace_eval = [
        {'8.8.8.8': ['192.168.1.1', '10.0.0.1', '8.8.8.8']},
        {'1.1.1.1': ['192.168.1.1', '10.0.0.1', '1.1.1.1']}
    ]

    # convert_to_list関数を呼び出す
    list_ping_eval, list_trace_eval = convert_to_list(df)

    # 結果が期待通りであることを確認
    assert list_ping_eval == expected_ping_eval
    assert list_trace_eval == expected_trace_eval
    
# テスト用のDataFrameを作成
@pytest.fixture
def sample_df():
    data = {
        'number': [1, 2, 3],
        'area': ['エリアA', 'エリアB', 'エリアC'],
        'name': ['拠点A', '拠点B', '拠点C'],
        'type': ['タイプ1', 'タイプ2', 'タイプ3']
    }
    return pd.DataFrame(data)

def test_select_kyoten_valid_input(sample_df):
    user_input = '1'  # 正しい入力
    with patch('builtins.input', return_value=user_input):
        selected_name, selected_type = select_kyoten(sample_df)
        
        assert selected_name == '拠点A'
        assert selected_type == 'タイプ1'

def test_select_kyoten_invalid_input(sample_df):
    user_inputs = ['4', 'invalid', '2']  # 最初は無効、次に無効、最後に有効な入力
    with patch('builtins.input', side_effect=user_inputs):
        selected_name, selected_type = select_kyoten(sample_df)
        
        assert selected_name == '拠点B'
        assert selected_type == 'タイプ2'

def test_select_kyoten_multiple_invalid_input(sample_df):
    user_inputs = ['invalid', 'invalid', '3']  # 全て無効、最後に有効な入力
    with patch('builtins.input', side_effect=user_inputs):
        selected_name, selected_type = select_kyoten(sample_df)
        
        assert selected_name == '拠点C'
        assert selected_type == 'タイプ3'

def test_select_test_type_large():
    selected_kyoten_type = '大規模'
    user_input = '1'  # 正しい入力
    with patch('builtins.input', return_value=user_input):
        selected_test_type = select_test_type(selected_kyoten_type)

        assert selected_test_type == '正常性試験'

def test_select_test_type_medium():
    selected_kyoten_type = '中規模'
    user_input = '2'  # 正しい入力
    with patch('builtins.input', return_value=user_input):
        selected_test_type = select_test_type(selected_kyoten_type)

        assert selected_test_type == '帯域保証網試験'

def test_select_test_type_small():
    selected_kyoten_type = '小規模'
    user_input = '3'  # 正しい入力
    with patch('builtins.input', return_value=user_input):
        selected_test_type = select_test_type(selected_kyoten_type)

        assert selected_test_type == '復旧試験'

def test_select_test_type_invalid_input():
    selected_kyoten_type = '大規模'
    user_inputs = ['invalid', '3']  # 最初は無効、次に有効な入力
    with patch('builtins.input', side_effect=user_inputs):
        selected_test_type = select_test_type(selected_kyoten_type)

        assert selected_test_type == 'ベストエフォート網試験'

def test_select_test_type_not_exist():
    selected_kyoten_type = '中規模'
    user_inputs = ['6', '1']  # 最初は存在しない番号、次に有効な番号
    with patch('builtins.input', side_effect=user_inputs):
        selected_test_type = select_test_type(selected_kyoten_type)

        assert selected_test_type == '正常性試験'
        
# テスト用のCSVデータを用意
TEST_CSV_DATA = """
kyoten_type,test_type,file_name
大規模,正常性試験,test_type_a.csv
大規模,帯域保証網試験,test_type_b.csv
中規模,正常性試験,test_type_c.csv
小規模,復旧試験,test_type_d.csv
"""

@pytest.fixture
def mock_test_info_csv(tmpdir):
    """テスト用のCSVファイルを一時的に作成するフィクスチャ"""
    test_info_path = tmpdir.join("test_info.csv")
    with open(test_info_path, 'w') as f:
        f.write(TEST_CSV_DATA)
    return str(test_info_path)

def test_get_test_type_path_valid(mock_test_info_csv):
    """有効な選択肢に対して正しいファイルパスを返すことをテスト"""
    result_1 = get_test_type_path("大規模", "正常性試験", mock_test_info_csv)
    assert result_1 == "../files/test_type_a.csv"
    result_2 = get_test_type_path("中規模", "正常性試験", mock_test_info_csv)
    assert result_2 == "../files/test_type_c.csv"
    
def test_get_test_type_path_invalid_type(mock_test_info_csv):
    """無効な拠点タイプに対して例外が発生することをテスト"""
    with pytest.raises(SelectionError, match="選択された拠点タイプまたは試験タイプが見つかりませんでした。"):
        get_test_type_path("無効なタイプ", "正常性試験", mock_test_info_csv)

def test_get_test_type_path_invalid_test(mock_test_info_csv):
    """無効な試験タイプに対して例外が発生することをテスト"""
    with pytest.raises(SelectionError, match="選択された拠点タイプまたは試験タイプが見つかりませんでした。"):
        get_test_type_path("大規模", "無効な試験", mock_test_info_csv)

def test_get_test_type_path_invalid_both(mock_test_info_csv):
    """無効な拠点タイプと試験タイプに対して例外が発生することをテスト"""
    with pytest.raises(SelectionError, match="選択された拠点タイプまたは試験タイプが見つかりませんでした。"):
        get_test_type_path("無効なタイプ", "無効な試験", mock_test_info_csv)
        
@mock.patch('os.path.exists')
def test_generate_unique_filename_first_call(mock_exists):
    """
    最初のファイル生成時のテスト: ファイルが存在しない場合、インクリメントされないことを確認。
    """
    mock_exists.return_value = False  # ファイルが存在しない場合
    
    results_dir = '../files'
    selected_kyoten_name = 'kyoten_name'
    selected_test_type = 'test_type'
    
    expected_filename = f'{results_dir}/{selected_kyoten_name}_{selected_test_type}_判定表.xlsx'
    generated_filename = generate_unique_filename(results_dir, selected_kyoten_name, selected_test_type)
    
    assert generated_filename == expected_filename

@mock.patch('os.path.exists')
def test_generate_unique_filename_second_call(mock_exists):
    """
    2回目のファイル生成時のテスト: 同じ名前のファイルが存在する場合、(1)が付くことを確認。
    """
    # 1回目は存在しないが、2回目は存在する
    mock_exists.side_effect = [True, False]
    
    results_dir = '../files'
    selected_kyoten_name = 'kyoten_name'
    selected_test_type = 'test_type'
    
    expected_filename = f'{results_dir}/{selected_kyoten_name}_{selected_test_type}_判定表(1).xlsx'
    generated_filename = generate_unique_filename(results_dir, selected_kyoten_name, selected_test_type)
    
    assert generated_filename == expected_filename

@mock.patch('os.path.exists')
def test_generate_unique_filename_third_call(mock_exists):
    """
    3回目のファイル生成時のテスト: 連続してファイルが存在する場合、(2)が付くことを確認。
    """
    # 1回目と2回目は存在し、3回目は存在しない
    mock_exists.side_effect = [True, True, False]
    
    results_dir = '../files'
    selected_kyoten_name = 'kyoten_name'
    selected_test_type = 'test_type'
    
    expected_filename = f'{results_dir}/{selected_kyoten_name}_{selected_test_type}_判定表(2).xlsx'
    generated_filename = generate_unique_filename(results_dir, selected_kyoten_name, selected_test_type)
    
    assert generated_filename == expected_filename
    
@pytest.fixture
def df():
    data = {
        'host': ['8.8.8.8', '8.8.4.4', '100.100.100.100'],
        'name': ['google_dns_1', 'google_dns_2', 'yahoo'],
        'data1': ['data_1', 'data_2', 'data_3'],
        'data2': ['data_1', 'data_2', 'data_3'],
        'data3': ['data_1', 'data_2', 'data_3'],
    }
    return pd.DataFrame(data)

# 正常なホスト名でのテストケース
def test_get_name_by_host_valid(df):
    assert get_name_by_host(df, '8.8.8.8') == 'google_dns_1'
    assert get_name_by_host(df, '100.100.100.100') == 'yahoo'

# 存在しないホスト名でのテストケース
def test_get_name_by_host_invalid(df):
    assert get_name_by_host(df, '1.1.1.1') is None

# 空のホスト名でのテストケース
def test_get_name_by_host_empty(df):
    assert get_name_by_host(df, '') is None

# Noneホスト名でのテストケース
def test_get_name_by_host_none(df):
    assert get_name_by_host(df, None) is None