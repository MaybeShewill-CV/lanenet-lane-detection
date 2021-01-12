/************************************************
* Author: MaybeShewill-CV
* File: configParser.h
* Date: 2019/10/10 上午10:39
************************************************/

#ifndef MNN_CONFIGPARSER_H
#define MNN_CONFIGPARSER_H

// Config parser

#include <exception>
#include <stdio.h>
#include <string>
#include <map>

const extern int __CONFIG_BUFFER_SIZE;

namespace beec {
    namespace config_parse_utils {

        class ConfigParser {

            using config_values = std::map<std::string, std::string>;
        public:
            explicit ConfigParser(const std::string& filename);

            ~ConfigParser() = default;

            config_values get_section(const std::string& section_name) const;

            config_values operator[](const std::string& section_name) const;

            void dump(FILE* log_file);

        private:
            std::map<std::string, std::map<std::string, std::string> > _m_sections;

            std::string read_header(const std::string& line);

            void read_configuration(const std::string& line, const std::string& header);

            // trim from start (in place)
            void ltrim(std::string &s);

            // trim from end (in place)
            void rtrim(std::string &s);

            // trim from both ends (in place)
            void trim(std::string &s);

            // trim from start (copying)
            std::string ltrim_copy(std::string s);

            // trim from end (copying)
            std::string rtrim_copy(std::string s);

            // trim from both ends (copying)
            std::string trim_copy(std::string s);
        };
    }
}

#endif //MNN_CONFIGPARSER_H
