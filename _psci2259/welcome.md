---
title: "PSCI 2259 Overview"
permalink: /psci2259/
lesson: 0
---
Welcome to the beginning of my class notes for PSCI 2259 Fall 2020 at Vanderbilt University taught by Professor Alan Wiseman. This course covers game theory from a poltical science perspective.

{% for lesson in site.psci2259 %}
  <h2>
    <a href="{{ lesson.url }}">{{ lesson.title }}</a>
  </h2>
{% endfor %}

