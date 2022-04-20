#!/usr/bin/env python
# -*- coding: utf-8 -*-

import mysql.connector


def main(config):
    db = mysql.connector.Connect(**config)
    cursor = db.cursor()
    
    cursor.execute("SELECT NOW()")
    output = cursor.fetchone()[0]
    cursor.close()
    db.close()
    
    return output


if __name__ == '__main__':
    #
    # Configure MySQL login and database to use in config.py
    #
    config = {
        "host": "192.168.0.41",
        "port": 3306,
        "user": "alpplas",
        "password": "1q2w3e4R",
        "auth_plugin": "mysql_native_password",
        "database": "",
        "charset": "utf8",
        "use_unicode": True,
        "get_warnings": True,
    }

    out = main(config)
    print(out)
