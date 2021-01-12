/************************************************
* Author: MaybeShewill-CV
* File: configParser.cpp
* Date: 2019/10/10 上午10:39
************************************************/

#include "config_parser.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace beec {
    namespace config_parse_utils {

        ConfigParser::ConfigParser(const std::string &filename) {

            std::ifstream fin(filename);

            if (fin.good()) {
                std::string line;
                std::string current_header = "";
                while (std::getline(fin, line)) {
                    trim(line);

                    // Skip empty lines
                    if (line.size() == 0)
                        continue;

                    switch (line[0]) {
                        case '#':
                        case ';':
                            // Ignore comments
                            break;
                        case '[':
                            // Section header
                            current_header = read_header(line);
                            break;
                        default:
                            // Everything else will be configurations
                            read_configuration(line, current_header);
                    }
                }
                fin.close();
            } else {
                throw std::runtime_error("File `" + filename + "` does not exist");
            }
        }

        std::map<std::string, std::string> ConfigParser::get_section(const std::string &section_name) const {

            if (_m_sections.count(section_name) == 0) {
                std::string error = "No such key: `" + section_name + "`";
                throw std::out_of_range(error);
            }
            return _m_sections.at(section_name);
        }

        std::map<std::string, std::string> ConfigParser::operator[](const std::string &section_name) const {

            if (_m_sections.count(section_name) == 0) {
                std::string error = "No such key: `" + section_name + "`";
                throw std::out_of_range(error);
            }
            return _m_sections.at(section_name);
        }

        void ConfigParser::dump(FILE *log_file) {

            // Set up iterators
            std::map<std::string, std::string>::iterator itr1;
            std::map<std::string, std::map<std::string, std::string> >::iterator itr2;
            for (itr2 = _m_sections.begin(); itr2 != _m_sections.end(); itr2++) {
                fprintf(log_file, "[%s]\n", itr2->first.c_str());
                for (itr1 = itr2->second.begin(); itr1 != itr2->second.end(); itr1++) {
                    fprintf(log_file, "%s=%s\n", itr1->first.c_str(), itr1->second.c_str());
                }
            }
        }

        std::string ConfigParser::read_header(const std::string &line) {

            if (line[line.size() - 1] != ']')
                throw std::runtime_error("Invalid section header: `" + line + "`");
            return trim_copy(line.substr(1, line.size() - 2));
        }

        void ConfigParser::read_configuration(const std::string &line, const std::string &header) {
            if (header == "") {
                std::string error = "No section provided for: `" + line + "`";
                throw std::runtime_error(error);
            }

            if (line.find('=') == std::string::npos) {
                std::string error = "Invalid configuration: `" + line + "`";
                throw std::runtime_error(error);
            }

            std::istringstream iss(line);
            std::string key;
            std::string val;
            std::getline(iss, key, '=');

            if (key.size() == 0) {
                std::string error = "No key found in configuration: `" + line + "`";
                throw std::runtime_error(error);
            }

            std::getline(iss, val);

            _m_sections[header][trim_copy(key)] = trim_copy(val);
        }

        // trim from start (in place)
        void ConfigParser::ltrim(std::string &s) {
            s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
                return !std::isspace(ch);
            }));
        }

        // trim from end (in place)
        void ConfigParser::rtrim(std::string &s) {
            s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
                return !std::isspace(ch);
            }).base(), s.end());
        }

        // trim from both ends (in place)
        void ConfigParser::trim(std::string &s) {
            ltrim(s);
            rtrim(s);
        }

        // trim from start (copying)
        std::string ConfigParser::ltrim_copy(std::string s) {
            ltrim(s);
            return s;
        }

        // trim from end (copying)
        std::string ConfigParser::rtrim_copy(std::string s) {
            rtrim(s);
            return s;
        }

        // trim from both ends (copying)
        std::string ConfigParser::trim_copy(std::string s) {
            trim(s);
            return s;
        }
    }
}