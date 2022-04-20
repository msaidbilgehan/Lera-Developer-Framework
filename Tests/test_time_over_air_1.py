# https://stackoverflow.com/questions/50557234/authentication-plugin-caching-sha2-password-is-not-supported


HOST="192.168.0.41"
USER="alpplas"
PASSWORD="1q2w3e4R"
AUTH_PLUGIN="mysql_native_password"


def request_Time_Over_MySQL_Database(host, user, password, auth_plugin, daytime_format="%Y-%m-%d_%H-%M-%S"):
    import mysql.connector
    
    # CONNECT TO DATABASE
    mysql_db_connection = mysql.connector.connect(
        host=host,
        user=user,
        password=password,
        auth_plugin=auth_plugin
    )
    mysql_db_cursor = mysql_db_connection.cursor()

    # EXECUTE AND FETCH RESULT
    mysql_db_cursor.execute("SELECT NOW()")
    db_time = mysql_db_cursor.fetchone()[0]

    # CLOSE CURSOR AND CONNECTION
    mysql_db_cursor.close()
    mysql_db_connection.close()
    
    return db_time.strftime(daytime_format)


# PRINT RESULTS
da = request_Time_Over_MySQL_Database(HOST, USER, PASSWORD, AUTH_PLUGIN)
print("HOST {0} TIME IS {1}".format(HOST, da))
