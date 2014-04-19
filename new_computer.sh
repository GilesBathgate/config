#!/bin/bash
apt-add-repository -y ppa:keepassx/daily
apt-get -y update 
apt-get -y dist-upgrade

#Stuff I dont want
apt-get -y purge nvidia-common
apt-get -y purge zeitgeist
apt-get -y purge .*unity.*
apt-get -y install unity-greeter gnome-session-fallback
apt-get -y install brasero brasero-cdrkit nautilus nautilus-sendto nautilus-share rhythmbox-plugin-cdrecorder shotwell 
apt-get -y purge .*ubuntuone.*
apt-get -y purge overlay-scrollbar
apt-get -y purge gnome-orca onboard
apt-get -y purge firefox
apt-get -y purge indicator-session
apt-get -y purge appmenu-qt5

#Stuff I need
apt-get -y install git keepassx pidgin chromium-browser vim ntp virtualbox meld gitg

#Rapcad dependencies
#apt-get -y install flex bison libcgal-dev libqt4-dev libdxflib-dev asciidoc source-highlight

#Sutff I want
#apt-get -y install qtcreator gcstar openssh-server picard transmission-gtk
#apt-get -y install libgnome2-0
